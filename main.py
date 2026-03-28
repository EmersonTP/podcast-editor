"""
Podcast Editor — FastAPI
Fluxo: upload vídeo → Whisper transcreve → Claude analisa e sugere cortes
       → humano aprova → FFmpeg executa → entregáveis prontos
"""

import json, os, uuid, time, subprocess, math, re, textwrap
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import aiofiles

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BRAND_NAME    = os.environ.get("BRAND_NAME",    "Podcast Editor")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")   # tiny/base/small/medium/large

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(BASE_DIR / "uploads")))
PROC_DIR   = Path(os.environ.get("PROC_DIR",   str(BASE_DIR / "processed")))
DATA_DIR   = Path(os.environ.get("DATA_DIR",   str(BASE_DIR / "data")))
JINGLE_IN  = Path(os.environ.get("JINGLE_IN",  str(BASE_DIR / "jingle_open.mp4")))
JINGLE_OUT = Path(os.environ.get("JINGLE_OUT", str(BASE_DIR / "jingle_close.mp4")))

for d in [UPLOAD_DIR, PROC_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Podcast Editor")

# ─── DB simples ───────────────────────────────────────────────────────────────
DB_FILE = DATA_DIR / "jobs.json"

def _load():
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text())
    return {}

def _save(db):
    DB_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2))

def _job(job_id):
    return _load().get(job_id)

def _update(job_id, **kwargs):
    db = _load()
    if job_id not in db:
        db[job_id] = {}
    db[job_id].update(kwargs)
    _save(db)

# ─── UTILS ────────────────────────────────────────────────────────────────────
def ts(seconds):
    """Float seconds → HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def ts_human(seconds):
    """Float seconds → MM:SS para exibição"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run_whisper(video_path: Path, job_id: str):
    """Transcreve com Whisper e salva resultado"""
    _update(job_id, status="transcribing", progress=10, msg="Transcrevendo com Whisper...")
    try:
        import whisper
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(
            str(video_path),
            language="pt",
            word_timestamps=True,
            verbose=False
        )
        # Montar segmentos com timestamps
        segments = []
        for seg in result["segments"]:
            segments.append({
                "id":    seg["id"],
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg["text"].strip(),
            })
        full_text = result["text"].strip()
        _update(job_id,
            status="analyzing",
            progress=40,
            msg="Transcrição concluída. Analisando com IA...",
            transcript=full_text,
            segments=segments,
        )
        return segments, full_text
    except Exception as e:
        _update(job_id, status="error", msg=f"Erro na transcrição: {e}")
        raise

def run_analysis(segments, full_text, job_id, duration):
    """Claude analisa e sugere cortes editoriais"""
    _update(job_id, status="analyzing", progress=55, msg="Claude analisando o conteúdo...")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        # Formatar transcrição com timestamps para o Claude
        transcript_with_ts = "\n".join(
            f"[{ts_human(s['start'])} → {ts_human(s['end'])}] {s['text']}"
            for s in segments
        )

        prompt = f"""Você é o melhor editor de podcast do mundo. Analise esta transcrição e forneça uma análise editorial completa.

DURAÇÃO TOTAL: {ts_human(duration)}

TRANSCRIÇÃO COMPLETA:
{transcript_with_ts}

Retorne um JSON válido com exatamente esta estrutura:
{{
  "episodio": {{
    "titulo_sugerido": "título principal do episódio",
    "titulos_alternativos": ["título 2", "título 3"],
    "resumo": "2-3 frases descrevendo o episódio",
    "descricao_plataformas": "descrição completa para Spotify/YouTube (150-200 palavras)",
    "tema_central": "qual é o tema central em uma frase"
  }},
  "cortes": [
    {{
      "id": 1,
      "tipo": "manter|cortar|comprimir",
      "inicio": 0.0,
      "fim": 0.0,
      "duracao": 0.0,
      "justificativa": "por que manter ou cortar",
      "energia": "alta|media|baixa",
      "prioridade": 1
    }}
  ],
  "melhor_clip": {{
    "inicio": 0.0,
    "fim": 0.0,
    "duracao": 0.0,
    "motivo": "por que este é o melhor clip para redes sociais"
  }},
  "frase_destaque": {{
    "texto": "a frase mais poderosa do episódio",
    "inicio": 0.0,
    "fim": 0.0
  }},
  "capitulos": [
    {{
      "inicio": 0.0,
      "titulo": "nome do capítulo"
    }}
  ],
  "problemas_detectados": [
    {{
      "tipo": "silencio_longo|vicio_linguagem|audio_ruim|repeticao",
      "inicio": 0.0,
      "fim": 0.0,
      "descricao": "descrição do problema"
    }}
  ],
  "estatisticas": {{
    "palavras_por_minuto": 0,
    "pausas_longas": 0,
    "energia_geral": "alta|media|baixa",
    "recomendacao_duracao_final": 0.0
  }}
}}

Regras:
- Sugira cortar: silêncios > 3 segundos, repetições, tangentes longas, vícios excessivos
- Sugira manter: argumentos fortes, histórias, dados, momentos de emoção, humor
- O clip para redes deve ter entre 45-90 segundos e ser o momento de MAIOR impacto
- Seja preciso nos timestamps — use os da transcrição
- Responda APENAS com o JSON, sem markdown, sem explicação"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        # Limpar possível markdown
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`").strip()

        analysis = json.loads(raw)
        _update(job_id,
            status="ready",
            progress=70,
            msg="Análise concluída. Revise os cortes sugeridos.",
            analysis=analysis,
        )
        return analysis
    except Exception as e:
        _update(job_id, status="error", msg=f"Erro na análise: {e}")
        raise

def run_export(job_id: str, approved_cuts, video_path: Path, make_clip: bool):
    """FFmpeg executa os cortes aprovados e monta o vídeo final"""
    _update(job_id, status="exporting", progress=75, msg="Executando cortes...")
    try:
        out_base = PROC_DIR / job_id
        out_base.mkdir(exist_ok=True)

        # ── 1. Montar lista de segmentos para manter ──────────────────────
        keeps = sorted(
            [c for c in approved_cuts if c["tipo"] == "manter"],
            key=lambda x: x["inicio"]
        )

        if not keeps:
            _update(job_id, status="error", msg="Nenhum segmento aprovado para manter.")
            return

        # ── 2. Criar concat list ──────────────────────────────────────────
        segment_files = []
        for i, cut in enumerate(keeps):
            seg_path = out_base / f"seg_{i:03d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(cut["inicio"]),
                "-to", str(cut["fim"]),
                "-i", str(video_path),
                "-c:v", "libx264", "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                str(seg_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            segment_files.append(seg_path)

        _update(job_id, progress=82, msg="Segmentos cortados. Montando vídeo...")

        # ── 3. Concatenar segmentos ───────────────────────────────────────
        concat_file = out_base / "concat.txt"
        with open(concat_file, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        body_path = out_base / "body.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy", str(body_path)
        ], capture_output=True, check=True)

        # ── 4. Adicionar vinheta se existir ───────────────────────────────
        final_path = out_base / "podcast_editado.mp4"
        parts = []
        if JINGLE_IN.exists():
            parts.append(str(JINGLE_IN))
        parts.append(str(body_path))
        if JINGLE_OUT.exists():
            parts.append(str(JINGLE_OUT))

        if len(parts) > 1:
            jingle_concat = out_base / "final_concat.txt"
            with open(jingle_concat, "w") as f:
                for p in parts:
                    f.write(f"file '{p}'\n")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(jingle_concat),
                "-c:v", "libx264", "-c:a", "aac",
                str(final_path)
            ], capture_output=True, check=True)
        else:
            final_path = body_path

        _update(job_id, progress=90, msg="Vídeo principal pronto. Gerando clip...")

        # ── 5. Gerar clip para redes sociais ──────────────────────────────
        clip_path = None
        job_data = _job(job_id)
        if make_clip and job_data and "analysis" in job_data:
            mc = job_data["analysis"].get("melhor_clip", {})
            if mc.get("inicio") is not None and mc.get("fim") is not None:
                clip_path = out_base / "clip_redes.mp4"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(mc["inicio"]),
                    "-to", str(mc["fim"]),
                    "-i", str(video_path),
                    "-c:v", "libx264", "-c:a", "aac",
                    str(clip_path)
                ], capture_output=True, check=True)

        # ── 6. Gerar transcrição TXT ──────────────────────────────────────
        transcript_path = out_base / "transcricao.txt"
        job_data = _job(job_id)
        if job_data and "transcript" in job_data:
            transcript_path.write_text(job_data["transcript"])

        _update(job_id,
            status="done",
            progress=100,
            msg="Tudo pronto!",
            output_video=str(final_path),
            output_clip=str(clip_path) if clip_path else None,
            output_transcript=str(transcript_path),
        )

    except subprocess.CalledProcessError as e:
        _update(job_id, status="error", msg=f"Erro FFmpeg: {e.stderr.decode()[:300]}")
    except Exception as e:
        _update(job_id, status="error", msg=f"Erro na exportação: {e}")

# ─── ROTAS ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(UI_HTML)

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{ext}"

    async with aiofiles.open(video_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Pegar duração via ffprobe
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ], capture_output=True, text=True)
        info = json.loads(result.stdout)
        duration = float(next(
            s["duration"] for s in info["streams"]
            if s.get("codec_type") == "video"
        ))
    except:
        duration = 0.0

    _update(job_id,
        status="uploaded",
        progress=5,
        msg="Arquivo recebido. Iniciando transcrição...",
        filename=file.filename,
        video_path=str(video_path),
        duration=duration,
        created_at=datetime.now().isoformat(),
    )

    background_tasks.add_task(process_job, job_id, video_path, duration)
    return JSONResponse({"job_id": job_id})

@app.post("/upload-jingle")
async def upload_jingle(type: str = Form(...), file: UploadFile = File(...)):
    """type: 'open' ou 'close'"""
    dest = JINGLE_IN if type == "open" else JINGLE_OUT
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)
    return JSONResponse({"ok": True, "type": type})

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = _job(job_id)
    if not job:
        return JSONResponse({"error": "Job não encontrado"}, status_code=404)
    # Retornar sem paths internos desnecessários
    safe = {k: v for k, v in job.items() if k not in ("video_path",)}
    return JSONResponse(safe)

@app.post("/approve/{job_id}")
async def approve(job_id: str, request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    approved_cuts = body.get("cuts", [])
    make_clip     = body.get("make_clip", True)

    job = _job(job_id)
    if not job:
        return JSONResponse({"error": "Job não encontrado"}, status_code=404)

    _update(job_id, approved_cuts=approved_cuts, status="exporting", progress=72,
            msg="Cortes aprovados. Exportando...")

    video_path = Path(job["video_path"])
    background_tasks.add_task(run_export, job_id, approved_cuts, video_path, make_clip)
    return JSONResponse({"ok": True})

@app.get("/download/{job_id}/{type}")
async def download(job_id: str, type: str):
    job = _job(job_id)
    if not job:
        return JSONResponse({"error": "Job não encontrado"}, status_code=404)

    paths = {
        "video":      job.get("output_video"),
        "clip":       job.get("output_clip"),
        "transcript": job.get("output_transcript"),
    }
    path = paths.get(type)
    if not path or not Path(path).exists():
        return JSONResponse({"error": "Arquivo não disponível"}, status_code=404)

    names = {
        "video":      "podcast_editado.mp4",
        "clip":       "clip_redes.mp4",
        "transcript": "transcricao.txt",
    }
    return FileResponse(path, filename=names.get(type, "download"))

# ─── BACKGROUND ───────────────────────────────────────────────────────────────

def process_job(job_id: str, video_path: Path, duration: float):
    try:
        segments, full_text = run_whisper(video_path, job_id)
        if ANTHROPIC_KEY:
            run_analysis(segments, full_text, job_id, duration)
        else:
            # Sem API key: gera análise básica apenas com timestamps
            _update(job_id,
                status="ready",
                progress=70,
                msg="Transcrição concluída. Configure ANTHROPIC_API_KEY para análise inteligente.",
                analysis=_basic_analysis(segments, duration)
            )
    except Exception as e:
        _update(job_id, status="error", msg=str(e))

def _basic_analysis(segments, duration):
    """Análise básica sem IA — divide em blocos de 5 minutos"""
    cuts = []
    block = 300  # 5 min
    i = 1
    for start in range(0, int(duration), block):
        end = min(start + block, duration)
        cuts.append({
            "id": i, "tipo": "manter",
            "inicio": float(start), "fim": float(end),
            "duracao": float(end - start),
            "justificativa": "Segmento automático — revise e ajuste",
            "energia": "media", "prioridade": i
        })
        i += 1
    mid = duration / 2
    return {
        "episodio": {
            "titulo_sugerido": "Título a definir",
            "titulos_alternativos": [],
            "resumo": "Análise manual necessária — configure ANTHROPIC_API_KEY",
            "descricao_plataformas": "",
            "tema_central": ""
        },
        "cortes": cuts,
        "melhor_clip": {"inicio": mid - 45, "fim": mid + 45, "duracao": 90, "motivo": "Trecho central do episódio"},
        "frase_destaque": {"texto": "", "inicio": 0, "fim": 0},
        "capitulos": [{"inicio": 0, "titulo": "Introdução"}],
        "problemas_detectados": [],
        "estatisticas": {
            "palavras_por_minuto": 0,
            "pausas_longas": 0,
            "energia_geral": "media",
            "recomendacao_duracao_final": duration * 0.85
        }
    }

# ─── UI HTML ──────────────────────────────────────────────────────────────────

UI_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Podcast Editor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap');
  :root{
    --red:#FC3D21;--black:#0A0A0A;--white:#FFFFFF;
    --gray1:#F5F5F5;--gray2:#E5E5E5;--gray3:#999;--gray4:#555;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--black);color:var(--white);font-family:'DM Sans',sans-serif;min-height:100vh}
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  /* NAV */
  nav{display:flex;align-items:center;gap:16px;padding:20px 48px;border-bottom:1px solid #1A1A1A;position:sticky;top:0;background:rgba(10,10,10,.95);backdrop-filter:blur(8px);z-index:100}
  .nav-logo{font-size:18px;font-weight:800;letter-spacing:-.03em}
  .nav-logo span{color:var(--red)}
  .nav-badge{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--gray3);border:1px solid #2A2A2A;padding:4px 10px;border-radius:2px}
  nav .spacer{flex:1}
  .jingle-btn{font-size:12px;color:var(--gray3);background:none;border:1px solid #2A2A2A;padding:8px 16px;cursor:pointer;font-family:inherit;transition:all .2s}
  .jingle-btn:hover{border-color:var(--gray3);color:var(--white)}

  /* MAIN */
  main{max-width:960px;margin:0 auto;padding:48px 24px}

  /* UPLOAD ZONE */
  .upload-zone{border:2px dashed #2A2A2A;padding:80px 40px;text-align:center;transition:all .3s;cursor:pointer;animation:fadeUp .6s both;position:relative}
  .upload-zone:hover,.upload-zone.drag{border-color:var(--red);background:rgba(252,61,33,.04)}
  .upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .upload-icon{font-size:48px;margin-bottom:24px;opacity:.4}
  .upload-title{font-size:24px;font-weight:800;letter-spacing:-.03em;margin-bottom:8px}
  .upload-sub{font-size:14px;color:var(--gray3)}
  .upload-sub strong{color:var(--white)}

  /* PROGRESS */
  #progress-section{display:none;animation:fadeUp .5s both}
  .progress-card{background:#111;border:1px solid #1E1E1E;padding:32px;margin-bottom:24px}
  .prog-header{display:flex;align-items:center;gap:16px;margin-bottom:24px}
  .prog-spinner{width:24px;height:24px;border:2px solid #2A2A2A;border-top-color:var(--red);border-radius:50%;animation:spin 1s linear infinite;flex-shrink:0}
  .prog-spinner.done{animation:none;border-color:var(--red);background:var(--red)}
  .prog-title{font-size:16px;font-weight:700}
  .prog-msg{font-size:13px;color:var(--gray3);margin-top:4px}
  .prog-bar-wrap{background:#1A1A1A;height:4px;border-radius:2px;overflow:hidden}
  .prog-bar{height:100%;background:var(--red);transition:width .5s ease;border-radius:2px}
  .prog-pct{font-size:12px;color:var(--gray3);text-align:right;margin-top:8px}

  /* ANALYSIS */
  #analysis-section{display:none;animation:fadeUp .5s both}
  .section-title{font-size:13px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--red);margin-bottom:20px}

  /* EPISODE META */
  .meta-card{background:#111;border:1px solid #1E1E1E;padding:28px;margin-bottom:20px}
  .meta-title-big{font-size:22px;font-weight:800;letter-spacing:-.03em;margin-bottom:8px}
  .meta-alts{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
  .meta-alt{font-size:12px;color:var(--gray3);border:1px solid #2A2A2A;padding:4px 12px;cursor:pointer;transition:all .2s}
  .meta-alt:hover{border-color:var(--red);color:var(--white)}
  .meta-body{font-size:14px;color:#888;line-height:1.7}
  .meta-desc{font-size:13px;color:#666;line-height:1.7;margin-top:12px;padding-top:12px;border-top:1px solid #1A1A1A}

  /* STATS BAR */
  .stats-bar{display:flex;gap:24px;background:#0D0D0D;border:1px solid #1A1A1A;padding:20px 28px;margin-bottom:20px}
  .stat{display:flex;flex-direction:column;gap:4px}
  .stat-val{font-size:20px;font-weight:800;color:var(--white)}
  .stat-val.red{color:var(--red)}
  .stat-lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--gray3)}

  /* CLIP HIGHLIGHT */
  .clip-card{background:rgba(252,61,33,.06);border:1px solid rgba(252,61,33,.2);padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:20px}
  .clip-icon{font-size:28px;flex-shrink:0}
  .clip-info{flex:1}
  .clip-title{font-size:14px;font-weight:700;margin-bottom:4px}
  .clip-sub{font-size:12px;color:var(--gray3)}
  .clip-ts{font-size:13px;font-weight:700;color:var(--red);font-variant-numeric:tabular-nums}

  /* FRASE DESTAQUE */
  .quote-card{background:#111;border-left:3px solid var(--red);padding:20px 24px;margin-bottom:20px}
  .quote-text{font-size:16px;font-weight:700;letter-spacing:-.02em;line-height:1.5;margin-bottom:8px}
  .quote-ts{font-size:12px;color:var(--gray3)}

  /* CORTES */
  .cuts-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
  .cuts-actions{display:flex;gap:8px}
  .btn-sm{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:8px 16px;border:1px solid #2A2A2A;background:none;color:var(--gray3);cursor:pointer;font-family:inherit;transition:all .2s}
  .btn-sm:hover{border-color:var(--white);color:var(--white)}
  .btn-sm.red{background:var(--red);border-color:var(--red);color:var(--white)}
  .btn-sm.red:hover{background:#E02E14}

  .cut-item{display:flex;align-items:stretch;gap:0;margin-bottom:8px;border:1px solid #1E1E1E;transition:border-color .2s}
  .cut-item:hover{border-color:#333}
  .cut-item.approved{border-color:rgba(72,199,142,.3)}
  .cut-item.removed{opacity:.35;border-color:#1A1A1A}
  .cut-indicator{width:4px;flex-shrink:0}
  .cut-indicator.manter{background:#48C78E}
  .cut-indicator.cortar{background:var(--red)}
  .cut-indicator.comprimir{background:#FFD166}
  .cut-body{flex:1;padding:16px 20px}
  .cut-top{display:flex;align-items:center;gap:12px;margin-bottom:6px}
  .cut-type{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:3px 8px}
  .cut-type.manter{background:rgba(72,199,142,.1);color:#48C78E}
  .cut-type.cortar{background:rgba(252,61,33,.1);color:var(--red)}
  .cut-type.comprimir{background:rgba(255,209,102,.1);color:#FFD166}
  .cut-ts{font-size:13px;font-weight:700;color:var(--white);font-variant-numeric:tabular-nums}
  .cut-dur{font-size:11px;color:var(--gray3)}
  .cut-energia{font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px 8px;border-radius:1px}
  .cut-energia.alta{background:rgba(252,61,33,.15);color:var(--red)}
  .cut-energia.media{background:rgba(255,255,255,.06);color:var(--gray3)}
  .cut-energia.baixa{background:rgba(255,255,255,.03);color:#444}
  .cut-just{font-size:13px;color:#888;line-height:1.5}
  .cut-actions{display:flex;align-items:center;gap:8px;padding:0 16px;flex-shrink:0}
  .cut-toggle{width:32px;height:32px;border:1px solid #2A2A2A;background:none;color:var(--gray3);cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .2s;flex-shrink:0}
  .cut-toggle:hover{border-color:var(--white);color:var(--white)}
  .cut-toggle.active{background:var(--red);border-color:var(--red);color:var(--white)}

  /* PROBLEMS */
  .problem-item{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#0D0D0D;border:1px solid #1A1A1A;margin-bottom:6px}
  .prob-icon{font-size:16px;flex-shrink:0}
  .prob-text{font-size:13px;color:#888;flex:1}
  .prob-ts{font-size:12px;color:var(--red);font-weight:700;flex-shrink:0}

  /* CAPITULOS */
  .chapter-item{display:flex;align-items:center;gap:16px;padding:10px 0;border-bottom:1px solid #1A1A1A}
  .chap-ts{font-size:13px;font-weight:700;color:var(--red);font-variant-numeric:tabular-nums;width:60px;flex-shrink:0}
  .chap-title{font-size:14px;color:var(--white)}

  /* EXPORT */
  .export-card{background:#111;border:1px solid #1E1E1E;padding:28px;margin-top:24px}
  .export-options{display:flex;align-items:center;gap:16px;margin-bottom:24px;flex-wrap:wrap}
  .checkbox-label{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:13px;color:#888}
  .checkbox-label input{accent-color:var(--red);width:15px;height:15px}
  .btn-export{width:100%;padding:18px;background:var(--red);color:var(--white);font-family:inherit;font-weight:800;font-size:13px;letter-spacing:.1em;text-transform:uppercase;border:none;cursor:pointer;transition:background .2s}
  .btn-export:hover{background:#E02E14}
  .btn-export:disabled{opacity:.4;cursor:not-allowed}

  /* DOWNLOADS */
  #downloads-section{display:none;animation:fadeUp .5s both}
  .downloads-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-top:20px}
  .dl-card{background:#111;border:1px solid #1E1E1E;padding:24px;text-align:center;transition:border-color .2s}
  .dl-card:hover{border-color:var(--red)}
  .dl-icon{font-size:32px;margin-bottom:12px}
  .dl-title{font-size:14px;font-weight:700;margin-bottom:4px}
  .dl-sub{font-size:12px;color:var(--gray3);margin-bottom:16px}
  .btn-dl{display:block;padding:10px;background:var(--red);color:var(--white);font-family:inherit;font-weight:700;font-size:11px;letter-spacing:.1em;text-transform:uppercase;border:none;cursor:pointer;text-decoration:none;transition:background .2s}
  .btn-dl:hover{background:#E02E14}

  /* JINGLE MODAL */
  .modal{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;display:none;align-items:center;justify-content:center;padding:24px;backdrop-filter:blur(4px)}
  .modal.open{display:flex}
  .modal-box{background:#111;border:1px solid #2A2A2A;max-width:480px;width:100%;padding:40px;animation:fadeUp .3s both}
  .modal-title{font-size:20px;font-weight:800;letter-spacing:-.03em;margin-bottom:8px}
  .modal-sub{font-size:13px;color:var(--gray3);margin-bottom:28px}
  .modal-close{float:right;background:none;border:none;color:var(--gray3);font-size:20px;cursor:pointer;margin-top:-8px}
  .upload-field{margin-bottom:20px}
  .upload-field label{display:block;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--gray3);margin-bottom:8px}
  .upload-field input[type=file]{width:100%;background:#0D0D0D;border:1px solid #2A2A2A;color:var(--white);padding:12px;font-family:inherit;font-size:13px;cursor:pointer}
  .btn-save{width:100%;padding:14px;background:var(--red);color:var(--white);font-family:inherit;font-weight:700;font-size:12px;letter-spacing:.1em;text-transform:uppercase;border:none;cursor:pointer;margin-top:8px}

  /* TRANSCRIPT */
  #transcript-section{display:none;animation:fadeUp .5s both;margin-top:24px}
  .transcript-box{background:#0D0D0D;border:1px solid #1A1A1A;padding:24px;max-height:320px;overflow-y:auto;font-size:13px;line-height:1.8;color:#888;white-space:pre-wrap}
  .transcript-box::-webkit-scrollbar{width:4px}
  .transcript-box::-webkit-scrollbar-thumb{background:var(--red)}
</style>
</head>
<body>

<nav>
  <div class="nav-logo">podcast<span>editor</span></div>
  <div class="nav-badge">Beta</div>
  <div class="spacer"></div>
  <button class="jingle-btn" onclick="document.getElementById('jingleModal').classList.add('open')">
    ⚡ Configurar Vinheta
  </button>
</nav>

<main>

  <!-- UPLOAD -->
  <div class="upload-zone" id="uploadZone">
    <input type="file" id="fileInput" accept="video/*,audio/*">
    <div class="upload-icon">🎙</div>
    <div class="upload-title">Arraste o vídeo do podcast aqui</div>
    <div class="upload-sub">ou clique para selecionar · <strong>MP4, MOV, MKV, M4A, MP3</strong> · até 2GB</div>
  </div>

  <!-- PROGRESS -->
  <div id="progress-section">
    <div class="progress-card">
      <div class="prog-header">
        <div class="prog-spinner" id="progSpinner"></div>
        <div>
          <div class="prog-title" id="progTitle">Processando...</div>
          <div class="prog-msg" id="progMsg">Aguarde</div>
        </div>
      </div>
      <div class="prog-bar-wrap"><div class="prog-bar" id="progBar" style="width:0%"></div></div>
      <div class="prog-pct" id="progPct">0%</div>
    </div>
  </div>

  <!-- ANÁLISE -->
  <div id="analysis-section">

    <!-- META DO EPISÓDIO -->
    <div class="section-title">📋 Episódio</div>
    <div class="meta-card" id="metaCard"></div>

    <!-- STATS -->
    <div class="stats-bar" id="statsBar"></div>

    <!-- MELHOR CLIP -->
    <div class="section-title">✂️ Melhor Clip para Redes</div>
    <div class="clip-card" id="clipCard"></div>

    <!-- FRASE DESTAQUE -->
    <div class="section-title">💬 Frase Destaque</div>
    <div class="quote-card" id="quoteCard"></div>

    <!-- CAPÍTULOS -->
    <div class="section-title">📑 Capítulos</div>
    <div id="chaptersContainer" style="margin-bottom:24px"></div>

    <!-- PROBLEMAS -->
    <div class="section-title" id="probTitle" style="display:none">⚠️ Problemas Detectados</div>
    <div id="problemsContainer" style="margin-bottom:24px"></div>

    <!-- CORTES -->
    <div class="cuts-header">
      <div class="section-title" style="margin:0">🔪 Cortes Sugeridos</div>
      <div class="cuts-actions">
        <button class="btn-sm" onclick="selectAll(true)">Aprovar todos</button>
        <button class="btn-sm" onclick="selectAll(false)">Remover todos</button>
      </div>
    </div>
    <div id="cutsContainer"></div>

    <!-- TRANSCRIÇÃO -->
    <div id="transcript-section">
      <div class="section-title">📝 Transcrição Completa</div>
      <div class="transcript-box" id="transcriptBox"></div>
    </div>

    <!-- EXPORTAR -->
    <div class="export-card">
      <div class="section-title" style="margin-bottom:16px">🚀 Exportar</div>
      <div class="export-options">
        <label class="checkbox-label">
          <input type="checkbox" id="makeClip" checked>
          Gerar clip de 60–90s para redes sociais
        </label>
        <label class="checkbox-label">
          <input type="checkbox" id="makeTranscript" checked>
          Exportar transcrição em texto
        </label>
      </div>
      <button class="btn-export" id="exportBtn" onclick="exportJob()">
        Executar cortes e exportar →
      </button>
    </div>
  </div>

  <!-- DOWNLOADS -->
  <div id="downloads-section">
    <div class="section-title">✅ Pronto para download</div>
    <div class="downloads-grid" id="downloadsGrid"></div>
  </div>

</main>

<!-- JINGLE MODAL -->
<div class="modal" id="jingleModal">
  <div class="modal-box">
    <button class="modal-close" onclick="document.getElementById('jingleModal').classList.remove('open')">×</button>
    <div class="modal-title">Configurar Vinheta</div>
    <div class="modal-sub">A vinheta é adicionada automaticamente em todos os episódios.</div>
    <div class="upload-field">
      <label>Vinheta de Abertura</label>
      <input type="file" id="jingleOpenFile" accept="video/*,audio/*">
    </div>
    <div class="upload-field">
      <label>Vinheta de Fechamento</label>
      <input type="file" id="jingleCloseFile" accept="video/*,audio/*">
    </div>
    <button class="btn-save" onclick="saveJingles()">Salvar vinhetas</button>
  </div>
</div>

<script>
let currentJobId = null;
let pollInterval = null;
let cutsData = [];

// ─── FORMAT ──────────────────────────────────────────────────────────────
function fmt(s) {
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

// ─── UPLOAD ──────────────────────────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

fileInput.addEventListener('change', e => { if (e.target.files[0]) startUpload(e.target.files[0]); });

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag');
  const f = e.dataTransfer.files[0]; if (f) startUpload(f);
});

async function startUpload(file) {
  uploadZone.style.display = 'none';
  document.getElementById('progress-section').style.display = 'block';
  updateProgress(5, 'Enviando arquivo...', `${file.name}`);

  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/upload', { method:'POST', body: fd });
  const data = await res.json();
  currentJobId = data.job_id;
  pollInterval = setInterval(pollStatus, 2000);
}

// ─── POLL ─────────────────────────────────────────────────────────────────
async function pollStatus() {
  if (!currentJobId) return;
  const res = await fetch(`/status/${currentJobId}`);
  const job = await res.json();

  updateProgress(job.progress || 0, job.msg || '...', '');

  if (job.status === 'ready') {
    clearInterval(pollInterval);
    document.getElementById('progSpinner').classList.add('done');
    showAnalysis(job);
  } else if (job.status === 'done') {
    clearInterval(pollInterval);
    document.getElementById('progSpinner').classList.add('done');
    showDownloads(job);
  } else if (job.status === 'error') {
    clearInterval(pollInterval);
    document.getElementById('progTitle').textContent = '❌ Erro';
    document.getElementById('progMsg').textContent = job.msg;
  }
}

function updateProgress(pct, title, msg) {
  document.getElementById('progTitle').textContent = title;
  document.getElementById('progMsg').textContent = msg;
  document.getElementById('progBar').style.width = pct + '%';
  document.getElementById('progPct').textContent = pct + '%';
}

// ─── ANALYSIS ────────────────────────────────────────────────────────────
function showAnalysis(job) {
  const a = job.analysis;
  if (!a) return;

  document.getElementById('analysis-section').style.display = 'block';

  // Meta
  const ep = a.episodio || {};
  document.getElementById('metaCard').innerHTML = `
    <div class="meta-title-big">${ep.titulo_sugerido || 'Sem título'}</div>
    <div class="meta-alts">
      ${(ep.titulos_alternativos||[]).map(t=>`<div class="meta-alt">${t}</div>`).join('')}
    </div>
    <div class="meta-body">${ep.resumo || ''}</div>
    ${ep.descricao_plataformas ? `<div class="meta-desc">${ep.descricao_plataformas}</div>` : ''}
  `;

  // Stats
  const st = a.estatisticas || {};
  const dur_min = Math.floor((job.duration||0)/60);
  const rec_min = Math.floor((st.recomendacao_duracao_final||0)/60);
  document.getElementById('statsBar').innerHTML = `
    <div class="stat"><div class="stat-val">${dur_min}min</div><div class="stat-lbl">Duração bruta</div></div>
    <div class="stat"><div class="stat-val red">${rec_min}min</div><div class="stat-lbl">Duração editada</div></div>
    <div class="stat"><div class="stat-val">${st.palavras_por_minuto||'—'}</div><div class="stat-lbl">Palavras/min</div></div>
    <div class="stat"><div class="stat-val">${st.pausas_longas||0}</div><div class="stat-lbl">Pausas longas</div></div>
    <div class="stat"><div class="stat-val">${(a.cortes||[]).filter(c=>c.tipo==='manter').length}</div><div class="stat-lbl">Segmentos mantidos</div></div>
    <div class="stat"><div class="stat-val red">${(a.cortes||[]).filter(c=>c.tipo==='cortar').length}</div><div class="stat-lbl">Segmentos cortados</div></div>
  `;

  // Clip
  const mc = a.melhor_clip || {};
  document.getElementById('clipCard').innerHTML = `
    <div class="clip-icon">🎬</div>
    <div class="clip-info">
      <div class="clip-title">Clip recomendado para redes sociais</div>
      <div class="clip-sub">${mc.motivo || ''}</div>
    </div>
    <div class="clip-ts">${fmt(mc.inicio||0)} → ${fmt(mc.fim||0)} · ${Math.round(mc.duracao||0)}s</div>
  `;

  // Quote
  const fq = a.frase_destaque || {};
  document.getElementById('quoteCard').innerHTML = fq.texto ? `
    <div class="quote-text">"${fq.texto}"</div>
    <div class="quote-ts">${fmt(fq.inicio||0)} → ${fmt(fq.fim||0)}</div>
  ` : '<div class="quote-text" style="color:#444">Nenhuma frase destaque detectada</div>';

  // Capítulos
  const caps = a.capitulos || [];
  document.getElementById('chaptersContainer').innerHTML = caps.map(c=>`
    <div class="chapter-item">
      <div class="chap-ts">${fmt(c.inicio||0)}</div>
      <div class="chap-title">${c.titulo}</div>
    </div>
  `).join('');

  // Problemas
  const probs = a.problemas_detectados || [];
  if (probs.length > 0) {
    document.getElementById('probTitle').style.display = 'block';
    const icons = { silencio_longo:'🔇', vicio_linguagem:'🗣', audio_ruim:'📢', repeticao:'🔄' };
    document.getElementById('problemsContainer').innerHTML = probs.map(p=>`
      <div class="problem-item">
        <div class="prob-icon">${icons[p.tipo]||'⚠️'}</div>
        <div class="prob-text">${p.descricao}</div>
        <div class="prob-ts">${fmt(p.inicio||0)}</div>
      </div>
    `).join('');
  }

  // Cortes
  cutsData = (a.cortes||[]).map(c=>({...c, approved: c.tipo==='manter'}));
  renderCuts();

  // Transcrição
  if (job.transcript) {
    document.getElementById('transcript-section').style.display = 'block';
    document.getElementById('transcriptBox').textContent = job.transcript;
  }
}

function renderCuts() {
  document.getElementById('cutsContainer').innerHTML = cutsData.map((c,i)=>`
    <div class="cut-item ${c.approved?'approved':''} ${!c.approved&&c.tipo==='manter'?'removed':''}" id="cut-${i}">
      <div class="cut-indicator ${c.tipo}"></div>
      <div class="cut-body">
        <div class="cut-top">
          <span class="cut-type ${c.tipo}">${c.tipo}</span>
          <span class="cut-ts">${fmt(c.inicio)} → ${fmt(c.fim)}</span>
          <span class="cut-dur">${Math.round(c.duracao||c.fim-c.inicio)}s</span>
          <span class="cut-energia ${c.energia||'media'}">${c.energia||'media'}</span>
        </div>
        <div class="cut-just">${c.justificativa||''}</div>
      </div>
      <div class="cut-actions">
        <button class="cut-toggle ${c.approved?'active':''}" onclick="toggleCut(${i})" title="${c.approved?'Remover':'Incluir'}">
          ${c.approved?'✓':'×'}
        </button>
      </div>
    </div>
  `).join('');
}

function toggleCut(i) {
  cutsData[i].approved = !cutsData[i].approved;
  renderCuts();
}

function selectAll(val) {
  cutsData = cutsData.map(c=>({...c, approved: val}));
  renderCuts();
}

// ─── EXPORT ───────────────────────────────────────────────────────────────
async function exportJob() {
  const approved = cutsData.filter(c=>c.approved);
  if (approved.length === 0) { alert('Selecione pelo menos um segmento para manter.'); return; }

  document.getElementById('exportBtn').disabled = true;
  document.getElementById('exportBtn').textContent = 'Exportando...';
  document.getElementById('analysis-section').style.display = 'none';
  document.getElementById('progress-section').style.display = 'block';
  updateProgress(72, 'Executando cortes...', 'FFmpeg processando os segmentos aprovados');

  await fetch(`/approve/${currentJobId}`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      cuts: approved,
      make_clip: document.getElementById('makeClip').checked
    })
  });

  pollInterval = setInterval(async ()=>{
    const res = await fetch(`/status/${currentJobId}`);
    const job = await res.json();
    updateProgress(job.progress||72, job.msg||'...', '');
    if (job.status==='done') {
      clearInterval(pollInterval);
      showDownloads(job);
    } else if (job.status==='error') {
      clearInterval(pollInterval);
      document.getElementById('progTitle').textContent = '❌ Erro';
      document.getElementById('progMsg').textContent = job.msg;
    }
  }, 2000);
}

// ─── DOWNLOADS ────────────────────────────────────────────────────────────
function showDownloads(job) {
  document.getElementById('progress-section').style.display = 'none';
  document.getElementById('downloads-section').style.display = 'block';

  const items = [
    { type:'video', icon:'🎬', title:'Podcast Editado', sub:'Vídeo final com vinheta', avail: !!job.output_video },
    { type:'clip',  icon:'📱', title:'Clip para Redes', sub:'60–90s do melhor momento', avail: !!job.output_clip },
    { type:'transcript', icon:'📝', title:'Transcrição', sub:'Texto completo do episódio', avail: !!job.output_transcript },
  ];

  document.getElementById('downloadsGrid').innerHTML = items.filter(i=>i.avail).map(i=>`
    <div class="dl-card">
      <div class="dl-icon">${i.icon}</div>
      <div class="dl-title">${i.title}</div>
      <div class="dl-sub">${i.sub}</div>
      <a class="btn-dl" href="/download/${currentJobId}/${i.type}" download>⬇ Baixar</a>
    </div>
  `).join('');
}

// ─── JINGLES ──────────────────────────────────────────────────────────────
async function saveJingles() {
  const openFile  = document.getElementById('jingleOpenFile').files[0];
  const closeFile = document.getElementById('jingleCloseFile').files[0];
  const uploads = [];
  if (openFile)  uploads.push(uploadJingle('open',  openFile));
  if (closeFile) uploads.push(uploadJingle('close', closeFile));
  await Promise.all(uploads);
  document.getElementById('jingleModal').classList.remove('open');
  alert('Vinhetas salvas com sucesso!');
}

async function uploadJingle(type, file) {
  const fd = new FormData();
  fd.append('type', type);
  fd.append('file', file);
  await fetch('/upload-jingle', { method:'POST', body: fd });
}
</script>
</body>
</html>"""
