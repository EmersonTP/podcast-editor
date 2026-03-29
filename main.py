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

def ts_srt(seconds):
    """Float seconds → HH:MM:SS,mmm para SRT"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(segments):
    """Converte segmentos do Whisper em string no formato SRT"""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{ts_srt(seg['start'])} --> {ts_srt(seg['end'])}")
        lines.append(seg['text'])
        lines.append("")
    return "\n".join(lines)

def generate_srt_clip(segments, clip_start, clip_end):
    """Gera SRT apenas para o trecho do clip, com timestamps relativos ao início do clip"""
    lines = []
    idx = 1
    for seg in segments:
        # só segmentos que estão dentro do clip
        if seg['end'] <= clip_start or seg['start'] >= clip_end:
            continue
        # ajusta timestamps para serem relativos ao início do clip
        start = max(seg['start'] - clip_start, 0)
        end   = min(seg['end']   - clip_start, clip_end - clip_start)
        lines.append(str(idx))
        lines.append(f"{ts_srt(start)} --> {ts_srt(end)}")
        lines.append(seg['text'])
        lines.append("")
        idx += 1
    return "\n".join(lines)

# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run_whisper(video_path: Path, job_id: str):
    """Transcreve com faster-whisper e salva resultado"""
    _update(job_id, status="transcribing", progress=10, msg="Transcrevendo com Whisper...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        raw_segments, _ = model.transcribe(
            str(video_path),
            language="pt",
            word_timestamps=True,
        )
        # Montar segmentos com timestamps
        segments = []
        texts = []
        for i, seg in enumerate(raw_segments):
            segments.append({
                "id":    i,
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text.strip(),
            })
            texts.append(seg.text.strip())
        full_text = " ".join(texts)
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

        # ── 5. Gerar clip horizontal + reel vertical ──────────────────────
        clip_path = None
        reel_path = None
        job_data = _job(job_id)
        if make_clip and job_data and "analysis" in job_data:
            mc = job_data["analysis"].get("melhor_clip", {})
            clip_start = mc.get("inicio")
            clip_end   = mc.get("fim")
            if clip_start is not None and clip_end is not None:

                # 5a. Clip horizontal (16:9)
                clip_path = out_base / "clip_redes.mp4"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(clip_start), "-to", str(clip_end),
                    "-i", str(video_path),
                    "-c:v", "libx264", "-c:a", "aac",
                    str(clip_path)
                ], capture_output=True, check=True)

                # 5b. Reel vertical 9:16 com fundo desfocado + legenda queimada
                _update(job_id, progress=94, msg="Gerando reel vertical 9:16...")
                clip_srt_path = out_base / "clip_legenda.srt"
                segs = job_data.get("segments", [])
                clip_srt_path.write_text(generate_srt_clip(segs, clip_start, clip_end))

                reel_nosub = out_base / "reel_nosub.mp4"
                # fundo: escala para cobrir 1080x1920 e desfoca; frente: escala proporcional centralizada
                vf_vertical = (
                    "[0:v]split=2[bg][fg];"
                    "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
                    "crop=1080:1920,boxblur=20:5[blurred];"
                    "[fg]scale=1080:-2:force_original_aspect_ratio=decrease[scaled];"
                    "[blurred][scaled]overlay=(W-w)/2:(H-h)/2[out]"
                )
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(clip_path),
                    "-filter_complex", vf_vertical,
                    "-map", "[out]", "-map", "0:a",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(reel_nosub)
                ], capture_output=True, check=True)

                # queimar legenda no reel
                reel_path = out_base / "reel.mp4"
                srt_escaped = str(clip_srt_path).replace("\\", "/").replace(":", "\\:")
                vf_sub = (
                    f"subtitles='{srt_escaped}':force_style='"
                    "FontSize=16,Bold=1,Alignment=2,"
                    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                    "Outline=2,Shadow=1,MarginV=60'"
                )
                result = subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(reel_nosub),
                    "-vf", vf_sub,
                    "-c:v", "libx264", "-c:a", "aac",
                    str(reel_path)
                ], capture_output=True)
                # se falhar na legenda, usa o reel sem legenda mesmo
                if result.returncode != 0:
                    reel_path = reel_nosub

        # ── 6. Gerar transcrição TXT e legenda SRT ────────────────────────
        job_data = _job(job_id)
        transcript_path = out_base / "transcricao.txt"
        srt_path = out_base / "legenda.srt"
        if job_data and "transcript" in job_data:
            transcript_path.write_text(job_data["transcript"])
        if job_data and "segments" in job_data:
            srt_path.write_text(generate_srt(job_data["segments"]))

        _update(job_id,
            status="done",
            progress=100,
            msg="Tudo pronto!",
            output_video=str(final_path),
            output_clip=str(clip_path) if clip_path else None,
            output_reel=str(reel_path) if reel_path else None,
            output_transcript=str(transcript_path),
            output_srt=str(srt_path),
        )

    except subprocess.CalledProcessError as e:
        _update(job_id, status="error", msg=f"Erro FFmpeg: {e.stderr.decode()[:300]}")
    except Exception as e:
        _update(job_id, status="error", msg=f"Erro na exportação: {e}")

# ─── ROTAS ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(UI_HTML)

@app.post("/upload-chunk")
async def upload_chunk(
    request: Request,
    job_id:      str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename:    str = Form(...),
    background_tasks: BackgroundTasks = None,
    chunk: UploadFile = File(...),
):
    """Recebe um chunk de arquivo. Quando todos chegarem, inicia o pipeline."""
    chunk_dir = UPLOAD_DIR / f"{job_id}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = chunk_dir / f"chunk_{chunk_index:05d}"
    async with aiofiles.open(chunk_path, "wb") as f:
        await f.write(await chunk.read())

    # Verificar se todos os chunks chegaram
    received = len(list(chunk_dir.glob("chunk_*")))
    if received < total_chunks:
        return JSONResponse({"ok": True, "received": received, "total": total_chunks})

    # Todos os chunks recebidos — montar arquivo final
    ext = Path(filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{ext}"

    async with aiofiles.open(video_path, "wb") as out:
        for i in range(total_chunks):
            cp = chunk_dir / f"chunk_{i:05d}"
            async with aiofiles.open(cp, "rb") as inp:
                await out.write(await inp.read())

    # Limpar chunks
    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    # Pegar duração via ffprobe
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ], capture_output=True, text=True)
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        # Tenta pegar de vídeo primeiro, depois áudio
        duration = 0.0
        for s in streams:
            d = s.get("duration")
            if d:
                duration = float(d)
                break
    except:
        duration = 0.0

    _update(job_id,
        status="uploaded",
        progress=8,
        msg="Arquivo recebido. Iniciando transcrição...",
        filename=filename,
        video_path=str(video_path),
        duration=duration,
        created_at=datetime.now().isoformat(),
    )

    background_tasks.add_task(process_job, job_id, video_path, duration)
    return JSONResponse({"ok": True, "assembled": True, "job_id": job_id})

@app.post("/upload-init")
async def upload_init(request: Request):
    """Cria job_id antes de começar os chunks."""
    body = await request.json()
    job_id = uuid.uuid4().hex[:12]
    _update(job_id,
        status="pending",
        progress=0,
        msg="Aguardando chunks...",
        filename=body.get("filename", ""),
        created_at=datetime.now().isoformat(),
    )
    return JSONResponse({"job_id": job_id})

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Fallback para upload simples (compatibilidade)."""
    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{ext}"

    async with aiofiles.open(video_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ], capture_output=True, text=True)
        info = json.loads(result.stdout)
        duration = float(next(
            (s["duration"] for s in info.get("streams", []) if s.get("duration")),
            0.0
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
        "reel":       job.get("output_reel"),
        "transcript": job.get("output_transcript"),
        "srt":        job.get("output_srt"),
    }
    path = paths.get(type)
    if not path or not Path(path).exists():
        return JSONResponse({"error": "Arquivo não disponível"}, status_code=404)

    names = {
        "video":      "podcast_editado.mp4",
        "clip":       "clip_redes.mp4",
        "reel":       "reel.mp4",
        "transcript": "transcricao.txt",
        "srt":        "legenda.srt",
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
<title>Podcast Editor — Emerson</title>
<style>
@font-face {
  font-family: 'Sohne';
  src: url('https://fonts.cdnfonts.com/s/15179/TestSohne-Buch.woff') format('woff');
  font-weight: 400; font-style: normal;
}
@font-face {
  font-family: 'Sohne';
  src: url('https://fonts.cdnfonts.com/s/15179/TestSohne-Halbfett.woff') format('woff');
  font-weight: 600; font-style: normal;
}
@font-face {
  font-family: 'Sohne';
  src: url('https://fonts.cdnfonts.com/s/15179/TestSohne-Dreiviertelfett.woff') format('woff');
  font-weight: 700; font-style: normal;
}
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --coral:   #FD7D59;
  --purple:  #924FAC;
  --black:   #080808;
  --ink:     #0D0D0D;
  --panel:   #111111;
  --border:  #1C1C1C;
  --muted:   #2A2A2A;
  --dim:     #555;
  --mid:     #888;
  --fog:     #AAAAAA;
  --white:   #F2F0EC;
  --green:   #3ECF8E;
  --yellow:  #F5C518;
  --sans: 'Sohne', 'DM Sans', sans-serif;
  --mono: 'IBM Plex Mono', monospace;
}

*, *::before, *::after { margin:0; padding:0; box-sizing:border-box }
html { scroll-behavior:smooth }
body {
  background: var(--black);
  color: var(--white);
  font-family: var(--sans);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}

/* NOISE */
body::before {
  content:'';
  position:fixed; inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events:none; z-index:9999; opacity:.5;
}

@keyframes fadeUp  { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes spin    { to{transform:rotate(360deg)} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:.25} }
@keyframes shimmer { from{background-position:-200% 0} to{background-position:200% 0} }

/* ─────────────────────────────────────────────
   LOGIN SCREEN
───────────────────────────────────────────── */
#login-screen {
  position:fixed; inset:0;
  background:var(--black);
  display:flex;
  align-items:center;
  justify-content:center;
  z-index:800;
  animation:fadeIn .4s both;
}

.login-bg {
  position:absolute; inset:0;
  background: radial-gradient(ellipse 60% 60% at 50% 40%, rgba(146,79,172,.07) 0%, transparent 70%);
  pointer-events:none;
}

.login-box {
  width:100%;
  max-width:400px;
  padding:0 24px;
  animation:fadeUp .6s .1s both;
  position:relative;
  z-index:1;
}

.login-logo {
  margin-bottom:40px;
  text-align:center;
}

.login-logo img {
  height:36px;
  width:auto;
  display:inline-block;
}

.login-heading {
  font-size:24px;
  font-weight:700;
  letter-spacing:-.03em;
  color:var(--white);
  margin-bottom:6px;
  text-align:center;
}

.login-sub {
  font-family:var(--mono);
  font-size:11px;
  letter-spacing:.1em;
  text-transform:uppercase;
  color:var(--dim);
  text-align:center;
  margin-bottom:36px;
}

.login-field {
  margin-bottom:14px;
}

.login-label {
  display:block;
  font-family:var(--mono);
  font-size:9px;
  letter-spacing:.18em;
  text-transform:uppercase;
  color:var(--dim);
  margin-bottom:8px;
}

.login-input {
  width:100%;
  background:#0C0C0C;
  border:1px solid var(--muted);
  color:var(--white);
  font-family:var(--sans);
  font-size:15px;
  padding:14px 16px;
  outline:none;
  transition:border-color .2s;
  -webkit-appearance:none;
}

.login-input:focus { border-color:var(--coral) }
.login-input::placeholder { color:var(--dim) }

.login-error {
  font-family:var(--mono);
  font-size:11px;
  color:#FF6B6B;
  text-align:center;
  min-height:18px;
  margin-bottom:8px;
}

.login-btn {
  width:100%;
  padding:16px;
  background:var(--coral);
  color:#fff;
  font-family:var(--sans);
  font-weight:700;
  font-size:14px;
  letter-spacing:.02em;
  border:none;
  cursor:pointer;
  margin-top:8px;
  transition:all .2s;
  position:relative;
  overflow:hidden;
}

.login-btn::after {
  content:'';
  position:absolute; inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);
  background-size:200% 100%;
  opacity:0;
  transition:opacity .2s;
}

.login-btn:hover { background:#e96a47 }
.login-btn:hover::after { opacity:1; animation:shimmer 1s ease infinite }

.login-divider {
  border:none;
  border-top:1px solid var(--border);
  margin:28px 0;
}

.login-footer {
  font-family:var(--mono);
  font-size:10px;
  color:var(--dim);
  text-align:center;
}

/* ─────────────────────────────────────────────
   APPROVAL SCREEN (aguardando autorização)
───────────────────────────────────────────── */
#approval-screen {
  position:fixed; inset:0;
  background:var(--black);
  display:none;
  align-items:center;
  justify-content:center;
  z-index:700;
  flex-direction:column;
  gap:0;
}

.approval-box {
  width:100%;
  max-width:480px;
  padding:0 24px;
  animation:fadeUp .6s both;
  text-align:center;
}

.approval-icon {
  width:64px; height:64px;
  border:1px solid var(--muted);
  border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  margin:0 auto 28px;
  font-size:24px;
  position:relative;
}

.approval-icon::after {
  content:'';
  position:absolute; inset:-6px;
  border-radius:50%;
  border:1px solid var(--border);
  animation:pulse 2s ease infinite;
}

.approval-heading {
  font-size:22px;
  font-weight:700;
  letter-spacing:-.03em;
  color:var(--white);
  margin-bottom:10px;
}

.approval-sub {
  font-size:14px;
  color:var(--mid);
  line-height:1.6;
  margin-bottom:32px;
}

.approval-user-badge {
  display:inline-flex;
  align-items:center;
  gap:10px;
  background:var(--panel);
  border:1px solid var(--border);
  padding:12px 20px;
  margin-bottom:32px;
}

.badge-avatar {
  width:32px; height:32px;
  border-radius:50%;
  background:var(--coral);
  display:flex; align-items:center; justify-content:center;
  font-size:13px;
  font-weight:700;
  color:#fff;
  flex-shrink:0;
}

.badge-info { text-align:left }

.badge-name {
  font-size:13px;
  font-weight:600;
  color:var(--white);
}

.badge-role {
  font-family:var(--mono);
  font-size:10px;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:var(--dim);
}

.approval-actions {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:8px;
}

.btn-approve {
  padding:14px 20px;
  background:var(--green);
  color:#000;
  font-family:var(--sans);
  font-weight:700;
  font-size:13px;
  border:none;
  cursor:pointer;
  transition:all .2s;
}

.btn-approve:hover { background:#35b87d }

.btn-deny {
  padding:14px 20px;
  background:none;
  color:var(--mid);
  font-family:var(--sans);
  font-weight:600;
  font-size:13px;
  border:1px solid var(--muted);
  cursor:pointer;
  transition:all .2s;
}

.btn-deny:hover { border-color:var(--fog); color:var(--white) }

.approval-pending-msg {
  font-family:var(--mono);
  font-size:11px;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:var(--dim);
  margin-top:20px;
  display:flex;
  align-items:center;
  justify-content:center;
  gap:8px;
}

.pending-dot {
  width:6px; height:6px;
  border-radius:50%;
  background:var(--yellow);
  animation:pulse 1.5s ease infinite;
}

/* ─────────────────────────────────────────────
   TOPBAR
───────────────────────────────────────────── */
#topbar {
  position:sticky; top:0; z-index:200;
  background:rgba(8,8,8,.92);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  display:flex; align-items:center;
  height:52px; padding:0;
}

.topbar-logo {
  display:flex; align-items:center;
  padding:0 24px;
  height:100%;
  border-right:1px solid var(--border);
  gap:10px;
}

.topbar-logo img { height:22px; width:auto }

.topbar-sep {
  width:1px; height:20px;
  background:var(--muted);
}

.topbar-app {
  font-family:var(--mono);
  font-size:10px;
  letter-spacing:.15em;
  text-transform:uppercase;
  color:var(--dim);
}

.topbar-spacer { flex:1 }

.topbar-user {
  display:flex; align-items:center;
  gap:10px;
  padding:0 20px;
  border-left:1px solid var(--border);
  height:100%;
  cursor:pointer;
}

.user-avatar {
  width:26px; height:26px;
  border-radius:50%;
  background:var(--coral);
  display:flex; align-items:center; justify-content:center;
  font-size:11px; font-weight:700; color:#fff;
  flex-shrink:0;
}

.user-name {
  font-size:12px;
  font-weight:600;
  color:var(--fog);
}

.btn-topbar {
  font-family:var(--mono);
  font-size:10px;
  letter-spacing:.1em;
  text-transform:uppercase;
  color:var(--mid);
  background:none;
  border:1px solid var(--muted);
  padding:6px 16px;
  cursor:pointer;
  margin-right:16px;
  transition:all .2s;
  white-space:nowrap;
}

.btn-topbar:hover { border-color:var(--coral); color:var(--white) }

.topbar-status {
  display:flex; align-items:center; gap:8px;
  padding:0 20px;
  border-left:1px solid var(--border);
  height:100%;
  font-family:var(--mono);
  font-size:10px;
  letter-spacing:.08em;
  color:var(--dim);
}

.status-dot {
  width:6px; height:6px;
  border-radius:50%;
  background:var(--dim);
  flex-shrink:0;
}

.status-dot.live { background:var(--green); animation:pulse 2s ease infinite }

/* ─────────────────────────────────────────────
   APP LAYOUT
───────────────────────────────────────────── */
#app { max-width:1080px; margin:0 auto; padding:0 40px 80px }

/* HERO */
#hero { padding:64px 0 48px; animation:fadeUp .7s both }

.hero-overline {
  font-family:var(--mono);
  font-size:10px; letter-spacing:.2em; text-transform:uppercase;
  color:var(--coral); margin-bottom:20px;
  display:flex; align-items:center; gap:12px;
}

.hero-overline::before {
  content:''; display:block;
  width:28px; height:1px; background:var(--coral);
}

.hero-headline {
  font-size:clamp(48px,6vw,76px);
  font-weight:700;
  letter-spacing:-.04em;
  line-height:.92;
  color:var(--white);
  margin-bottom:24px;
}

.hero-headline span { color:var(--coral) }

.hero-sub {
  font-size:15px; color:var(--mid);
  line-height:1.65; max-width:500px;
  margin-bottom:48px;
}

/* UPLOAD ZONE */
#upload-zone {
  position:relative;
  border:1px solid var(--muted);
  background:var(--panel);
  padding:72px 48px;
  text-align:center;
  cursor:pointer;
  transition:all .3s ease;
  overflow:hidden;
}

#upload-zone::before {
  content:'';
  position:absolute; inset:0;
  background:linear-gradient(135deg,rgba(253,125,89,.07) 0%,transparent 60%);
  opacity:0; transition:opacity .3s;
}

#upload-zone:hover, #upload-zone.drag {
  border-color:var(--coral);
  box-shadow:0 0 0 1px var(--coral), 0 24px 48px rgba(253,125,89,.06);
  transform:translateY(-1px);
}

#upload-zone:hover::before, #upload-zone.drag::before { opacity:1 }

#upload-zone input[type=file] {
  position:absolute; inset:0;
  opacity:0; cursor:pointer;
  width:100%; height:100%;
}

.wave-viz {
  display:flex; align-items:center; justify-content:center;
  gap:3px; height:52px; margin-bottom:28px;
}

.w { width:3px; border-radius:2px; background:var(--muted); transition:background .3s }
#upload-zone:hover .w { background:var(--coral) }

.w:nth-child(1){height:8px}.w:nth-child(2){height:22px}.w:nth-child(3){height:38px}
.w:nth-child(4){height:28px}.w:nth-child(5){height:52px}.w:nth-child(6){height:44px}
.w:nth-child(7){height:52px}.w:nth-child(8){height:38px}.w:nth-child(9){height:30px}
.w:nth-child(10){height:44px}.w:nth-child(11){height:22px}.w:nth-child(12){height:14px}
.w:nth-child(13){height:8px}

.upload-cta {
  font-size:20px; font-weight:700;
  letter-spacing:-.03em; color:var(--white);
  margin-bottom:8px;
}

.upload-hint {
  font-family:var(--mono); font-size:11px;
  letter-spacing:.1em; text-transform:uppercase;
  color:var(--dim);
}

.upload-hint strong { color:var(--fog) }

/* PROCESSING */
#processing-panel { display:none; padding:56px 0; animation:fadeIn .4s both }

.proc-wrap {
  display:grid;
  grid-template-columns:88px 1fr;
  gap:32px; align-items:start;
  margin-bottom:40px;
}

.proc-num {
  font-size:88px; font-weight:700;
  letter-spacing:-.06em; line-height:1;
  color:var(--border);
  transition:color .5s; user-select:none;
}

.proc-num.lit { color:var(--coral) }

.proc-meta { padding-top:8px }

.proc-stage {
  font-family:var(--mono);
  font-size:9px; letter-spacing:.2em; text-transform:uppercase;
  color:var(--coral); margin-bottom:10px;
}

.proc-title {
  font-size:28px; font-weight:700;
  letter-spacing:-.03em; color:var(--white);
  margin-bottom:6px; line-height:1.1;
}

.proc-msg { font-size:13px; color:var(--mid); line-height:1.5 }

.prog-track {
  background:var(--border); height:1px;
  position:relative; overflow:hidden;
  margin-bottom:10px;
}

.prog-fill {
  height:100%; background:var(--coral);
  transition:width .6s ease; position:relative;
}

.prog-fill::after {
  content:''; position:absolute;
  right:0; top:-1px;
  width:48px; height:3px;
  background:linear-gradient(90deg,transparent,rgba(253,125,89,.5));
}

.prog-meta-row {
  display:flex; justify-content:space-between;
  font-family:var(--mono); font-size:10px; color:var(--dim);
}

.steps-list { border:1px solid var(--border); margin-top:28px }

.step-item {
  display:flex; align-items:center; gap:18px;
  padding:14px 22px;
  border-bottom:1px solid var(--border);
  transition:background .2s;
}

.step-item:last-child { border-bottom:none }
.step-item.active { background:rgba(253,125,89,.03) }
.step-item.done   { opacity:.45 }

.step-ico {
  width:26px; height:26px;
  border:1px solid var(--muted); border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-family:var(--mono); font-size:9px; color:var(--dim);
  flex-shrink:0; transition:all .3s;
}

.step-ico.spinning { border-color:var(--coral); color:var(--coral); animation:spin 1.2s linear infinite }
.step-ico.ok       { border-color:var(--green); color:var(--green); animation:none }

.step-lbl { font-size:13px; color:var(--mid); flex:1 }
.step-item.active .step-lbl { color:var(--white) }

/* DIVIDER */
.sec-div {
  display:flex; align-items:center; gap:14px;
  margin:44px 0 24px;
}

.sec-n { font-family:var(--mono); font-size:12px; color:var(--coral); min-width:20px }
.sec-lbl { font-family:var(--mono); font-size:9px; letter-spacing:.2em; text-transform:uppercase; color:var(--dim); white-space:nowrap }
.sec-line { flex:1; height:1px; background:var(--border) }

/* ANALYSIS */
#analysis-section { display:none; animation:fadeUp .6s both }

/* Episode block */
.ep-grid {
  display:grid;
  grid-template-columns:1fr 300px;
  gap:2px; margin-bottom:2px;
}

.ep-main {
  background:var(--panel); border:1px solid var(--border);
  padding:32px 36px;
}

.ep-eyebrow {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.2em; text-transform:uppercase;
  color:var(--coral); margin-bottom:14px;
}

.ep-title {
  font-size:clamp(20px,2.5vw,30px);
  font-weight:700; letter-spacing:-.03em;
  line-height:1.1; color:var(--white);
  margin-bottom:14px;
}

.ep-alts {
  display:flex; flex-wrap:wrap; gap:6px; margin-bottom:18px;
}

.ep-alt {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.06em; color:var(--dim);
  border:1px solid var(--muted); padding:4px 10px;
  cursor:pointer; transition:all .2s;
}

.ep-alt:hover { border-color:var(--fog); color:var(--fog) }

.ep-summary {
  font-size:13px; color:var(--mid); line-height:1.7;
  border-top:1px solid var(--border); padding-top:14px;
}

/* Stats panel */
.stats-panel {
  background:var(--coral); border:1px solid var(--coral);
  padding:32px 28px;
  display:flex; flex-direction:column; gap:0;
}

.stats-head {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.2em; text-transform:uppercase;
  color:rgba(0,0,0,.4); margin-bottom:20px;
}

.stat-row {
  padding:14px 0;
  border-bottom:1px solid rgba(0,0,0,.12);
}

.stat-row:last-of-type { border-bottom:none }

.stat-val {
  font-size:32px; font-weight:700;
  letter-spacing:-.04em; color:#fff;
  line-height:1;
}

.stat-lbl {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.12em; text-transform:uppercase;
  color:rgba(0,0,0,.45);
}

/* Highlights */
.hl-grid {
  display:grid; grid-template-columns:1fr 1fr;
  gap:2px; margin-bottom:2px;
}

.hl-card {
  background:var(--panel); border:1px solid var(--border);
  padding:26px 30px; transition:border-color .2s;
}

.hl-card:hover { border-color:var(--muted) }

.hl-eye {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.18em; text-transform:uppercase;
  color:var(--coral); margin-bottom:14px;
  display:flex; align-items:center; gap:8px;
}

.hl-eye::before { content:''; display:block; width:14px; height:1px; background:var(--coral) }

.hl-time {
  font-family:var(--mono); font-size:26px;
  font-weight:500; letter-spacing:-.01em;
  color:var(--white); margin-bottom:8px;
}

.hl-desc { font-size:12px; color:var(--mid); line-height:1.6 }

.hl-quote {
  font-size:18px; font-weight:700;
  letter-spacing:-.03em; line-height:1.4;
  color:var(--white); margin-bottom:8px;
}

.hl-quote::before { content:'C'; color:var(--coral) }
.hl-quote::after  { content:'D'; color:var(--coral) }

/* Chapters */
.chaps-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
  gap:2px; margin-bottom:2px;
}

.chap-card {
  background:var(--panel); border:1px solid var(--border);
  padding:18px 22px; display:flex; flex-direction:column; gap:5px;
}

.chap-ts {
  font-family:var(--mono); font-size:18px;
  font-weight:500; color:var(--coral);
}

.chap-title { font-size:12px; color:var(--mid); line-height:1.4 }

/* Problems */
.prob-row {
  display:flex; align-items:center; gap:18px;
  padding:12px 22px;
  background:var(--panel); border:1px solid var(--border);
  border-bottom:none; transition:background .2s;
}

.prob-row:last-child { border-bottom:1px solid var(--border) }
.prob-row:hover { background:#141414 }

.prob-type {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.12em; text-transform:uppercase;
  color:var(--yellow); min-width:120px;
}

.prob-desc { font-size:12px; color:var(--mid); flex:1; line-height:1.4 }
.prob-at   { font-family:var(--mono); font-size:11px; font-weight:500; color:var(--yellow) }

/* Cuts table */
.cuts-bar {
  display:flex; align-items:center;
  justify-content:space-between;
  padding:10px 0;
  border-bottom:1px solid var(--border);
  margin-bottom:2px;
}

.cuts-info { font-family:var(--mono); font-size:11px; color:var(--dim) }
.cuts-info strong { color:var(--white) }
.cuts-btns { display:flex; gap:8px }

.btn-sm {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.1em; text-transform:uppercase;
  color:var(--dim); background:none;
  border:1px solid var(--muted);
  padding:6px 12px; cursor:pointer; transition:all .2s;
}

.btn-sm:hover { border-color:var(--fog); color:var(--white) }

.cut-row {
  display:grid;
  grid-template-columns:3px 110px 1fr 56px 72px 42px;
  align-items:stretch;
  background:var(--panel); border:1px solid var(--border);
  border-bottom:none; transition:background .2s;
}

.cut-row:last-child { border-bottom:1px solid var(--border) }
.cut-row:hover { background:#131313 }
.cut-row.removed { opacity:.35 }

.cut-bar { width:3px; flex-shrink:0 }
.cut-bar.manter    { background:var(--green) }
.cut-bar.cortar    { background:var(--coral) }
.cut-bar.comprimir { background:var(--yellow) }
.cut-bar.removed   { background:var(--border) }

.cut-tc {
  padding:16px 18px;
  display:flex; flex-direction:column; justify-content:center; gap:3px;
  border-right:1px solid var(--border);
}

.tc-r {
  font-family:var(--mono); font-size:12px;
  font-weight:500; color:var(--white);
}

.tc-d { font-family:var(--mono); font-size:10px; color:var(--dim) }

.cut-body {
  padding:16px 18px;
  display:flex; flex-direction:column; justify-content:center; gap:5px;
}

.cut-tags { display:flex; gap:5px; align-items:center }

.ctag {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.1em; text-transform:uppercase;
  padding:2px 7px;
}

.ctag.manter    { background:rgba(62,207,142,.07);  color:var(--green);  border:1px solid rgba(62,207,142,.15) }
.ctag.cortar    { background:rgba(253,125,89,.07);  color:var(--coral);  border:1px solid rgba(253,125,89,.15) }
.ctag.comprimir { background:rgba(245,197,24,.07);  color:var(--yellow); border:1px solid rgba(245,197,24,.15) }
.ctag.alta      { background:rgba(253,125,89,.05);  color:var(--coral);  border:1px solid rgba(253,125,89,.1) }
.ctag.media     { background:rgba(255,255,255,.03); color:var(--dim);    border:1px solid var(--border) }
.ctag.baixa     { background:transparent; color:#2A2A2A; border:1px solid #181818 }

.cut-reason { font-size:12px; color:var(--dim); line-height:1.5 }

.cut-prio {
  padding:16px 10px;
  display:flex; align-items:center; justify-content:center;
  border-right:1px solid var(--border);
  border-left:1px solid var(--border);
}

.pip { width:7px; height:7px; border-radius:50% }
.p1 { background:var(--coral); box-shadow:0 0 6px rgba(253,125,89,.5) }
.p2 { background:var(--yellow) }
.p3 { background:var(--dim) }

.cut-save {
  padding:16px 12px;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  border-right:1px solid var(--border);
}

.save-v { font-family:var(--mono); font-size:12px; color:var(--green) }
.save-l { font-family:var(--mono); font-size:9px; color:var(--dim); letter-spacing:.06em }

.cut-tog-cell {
  display:flex; align-items:center;
  justify-content:center; padding:0 6px;
}

.cut-toggle {
  width:26px; height:26px;
  border:1px solid var(--muted); background:none;
  color:var(--dim); cursor:pointer;
  font-size:12px; font-family:var(--mono);
  display:flex; align-items:center; justify-content:center;
  transition:all .15s;
}

.cut-toggle:hover { border-color:var(--white); color:var(--white) }
.cut-toggle.on    { background:var(--green); border-color:var(--green); color:#000 }

/* Transcript */
.tx-wrap {
  background:var(--panel); border:1px solid var(--border);
}

.tx-inner {
  font-family:var(--mono); font-size:12px;
  line-height:1.9; color:var(--mid);
  padding:26px 30px; max-height:260px;
  overflow-y:auto; white-space:pre-wrap;
}

.tx-inner::-webkit-scrollbar { width:3px }
.tx-inner::-webkit-scrollbar-thumb { background:var(--muted) }

/* Export */
.export-panel {
  background:var(--panel); border:1px solid var(--border);
  padding:32px 36px; margin-top:2px;
}

.exp-opts { display:flex; gap:28px; margin-bottom:24px; flex-wrap:wrap }

.opt-lbl {
  display:flex; align-items:center; gap:9px;
  cursor:pointer;
  font-family:var(--mono); font-size:10px;
  letter-spacing:.08em; text-transform:uppercase;
  color:var(--mid); transition:color .2s;
}

.opt-lbl:hover { color:var(--white) }
.opt-lbl input[type=checkbox] { accent-color:var(--coral); width:13px; height:13px }

.btn-export {
  width:100%; padding:18px;
  background:var(--coral); color:#fff;
  font-family:var(--sans); font-weight:700;
  font-size:13px; letter-spacing:.04em;
  border:none; cursor:pointer; transition:all .2s;
  position:relative; overflow:hidden;
}

.btn-export::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);
  transform:translateX(-100%); transition:transform .4s;
}

.btn-export:hover { background:#e96a47 }
.btn-export:hover::after { transform:translateX(100%) }
.btn-export:disabled { opacity:.3; cursor:not-allowed }

/* Downloads */
#downloads-section { display:none; animation:fadeUp .5s both; padding:56px 0 }

.dl-grid {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:2px;
}

.dl-card {
  background:var(--panel); border:1px solid var(--border);
  padding:32px 28px; transition:border-color .2s;
}

.dl-card:hover { border-color:var(--muted) }

.dl-n {
  font-size:64px; font-weight:700;
  letter-spacing:-.06em; line-height:1;
  color:var(--border); margin-bottom:18px;
  transition:color .3s;
}

.dl-card:hover .dl-n { color:var(--muted) }
.dl-ico { font-size:22px; margin-bottom:10px }
.dl-title { font-size:18px; font-weight:700; letter-spacing:-.03em; color:var(--white); margin-bottom:4px }
.dl-sub { font-family:var(--mono); font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--dim); margin-bottom:22px; flex:1 }

.btn-dl {
  display:block; text-align:center; padding:11px 18px;
  background:var(--coral); color:#fff;
  font-family:var(--mono); font-size:10px; font-weight:500;
  letter-spacing:.12em; text-transform:uppercase;
  text-decoration:none; border:none;
  cursor:pointer; transition:background .2s;
}

.btn-dl:hover { background:#e96a47 }

/* Jingle Modal */
.modal-ov {
  position:fixed; inset:0;
  background:rgba(0,0,0,.9); z-index:500;
  display:none; align-items:center; justify-content:center;
  padding:24px; backdrop-filter:blur(8px);
}

.modal-ov.open { display:flex }

.modal-box {
  background:var(--ink); border:1px solid var(--muted);
  max-width:480px; width:100%;
  animation:fadeUp .3s both;
}

.modal-hd {
  padding:26px 32px 22px;
  border-bottom:1px solid var(--border);
  display:flex; align-items:flex-start;
  justify-content:space-between; gap:16px;
}

.modal-title {
  font-size:20px; font-weight:700;
  letter-spacing:-.03em; color:var(--white);
}

.modal-sub {
  font-family:var(--mono); font-size:9px;
  letter-spacing:.1em; text-transform:uppercase;
  color:var(--dim); margin-top:5px;
}

.modal-close {
  background:none; border:none; color:var(--dim);
  font-size:18px; cursor:pointer; transition:color .2s;
}

.modal-close:hover { color:var(--white) }
.modal-bd { padding:26px 32px 32px }

.f-field { margin-bottom:16px }

.f-label {
  display:block;
  font-family:var(--mono); font-size:9px;
  letter-spacing:.18em; text-transform:uppercase;
  color:var(--dim); margin-bottom:7px;
}

.f-row {
  border:1px solid var(--muted); background:#0A0A0A;
  padding:11px 14px;
  display:flex; align-items:center; gap:10px;
  cursor:pointer; transition:border-color .2s;
}

.f-row:hover { border-color:var(--fog) }

.f-row input[type=file] { position:absolute; opacity:0; pointer-events:none }

.f-name { font-family:var(--mono); font-size:11px; color:var(--mid); flex:1 }
.f-browse { font-family:var(--mono); font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--dim); border:1px solid var(--muted); padding:3px 9px; white-space:nowrap }

.btn-save {
  width:100%; padding:14px;
  background:var(--coral); color:#fff;
  font-family:var(--sans); font-weight:700;
  font-size:13px; border:none; cursor:pointer;
  margin-top:6px; transition:background .2s;
}

.btn-save:hover { background:#e96a47 }

@media(max-width:768px) {
  #app { padding:0 20px 60px }
  .ep-grid { grid-template-columns:1fr }
  .hl-grid { grid-template-columns:1fr }
  .dl-grid { grid-template-columns:1fr }
  .cut-row { grid-template-columns:3px 90px 1fr 42px }
  .cut-prio,.cut-save { display:none }
}
</style>
</head>
<body>

<!-- ══ LOGIN SCREEN ══════════════════════════════════════════ -->
<div id="login-screen">
  <div class="login-bg"></div>
  <div class="login-box">
    <div class="login-logo">
      <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj48c3ZnIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIHZpZXdCb3g9IjAgMCAyNjQgMTQ0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHhtbDpzcGFjZT0icHJlc2VydmUiIHhtbG5zOnNlcmlmPSJodHRwOi8vd3d3LnNlcmlmLmNvbS8iIHN0eWxlPSJmaWxsLXJ1bGU6ZXZlbm9kZDtjbGlwLXJ1bGU6ZXZlbm9kZDtzdHJva2UtbGluZWpvaW46cm91bmQ7c3Ryb2tlLW1pdGVybGltaXQ6MjsiPjxwYXRoIGQ9Ik0yNDcuMTIyLDExMy41OTZjLTAuNjE5LDAuMDExIC0wLjkzMywtMC4wMTUgLTEuNTU2LDAuMTUyYy0xLjYxNCwwLjQzNCAtMi42NjUsMC44NzQgLTQuMDI4LDIuODE0Yy0xLjExLDEuNTggLTEuNjQyLDQuNjAzIC0xLjcwMiw1LjI3NGMtMC40MTQsNC42MyAtMC4wNjIsMTEuNDAyIC0wLjA1OSwxMS41MmMwLjA1NCwyLjQwOSAwLjE4NiwyLjM5NiAwLjMxMSwzLjc2M2MwLjM5MSw0LjI4IDAuNDA3LDQuMzA2IDAuMjA5LDQuNzI2Yy0wLjI3NCwwLjU4IC0xLjEwMSwwLjUxOSAtNC4xNDgsMC41MTljLTYuMzczLDAuMDAxIC02LjQzOSwwLjA0IC02LjgzOSwtMC4zMDVjLTAuODIzLC0wLjcwNyAtMC40ODMsLTEuNzIxIC0wLjM3NywtMi40NTZjMC4wNDQsLTAuMzAxIDAuNjc1LC0xMC4zNjIgMC43MjYsLTE1LjYzMmMwLjA3NCwtNy43NjQgLTAuMDM2LC05LjIwOCAtMC4wOTcsLTEwLjAwNmMtMC4yNDIsLTMuMTcyIC0wLjIyNywtMy4xNyAtMC4yNTUsLTMuNDQ1Yy0wLjAyNiwtMC4yNSAtMC4yODQsLTIuNzExIC0wLjM1NiwtMy4xMjNjLTAuMjQyLC0xLjM2OCAtMC4zMTUsLTEuODY2IDEuOTA0LC0yLjQwMWMxLjQwNywtMC4zMzkgMS4zOCwtMC40NDEgMi44MDcsLTAuNjU3YzAuNTMzLC0wLjA4MSAxLjA1NSwtMC4zNjIgMi44MTYsLTAuNTk0YzAuOTc0LC0wLjEyOCAzLjIzMSwtMS4wODcgMy4xODEsMS40NjVjLTAuMDQ1LDIuMjM4IC0wLjU0LDQuMTcgLTAuNTY2LDQuNDdjLTAuMDQ2LDAuNTM4IC0wLjEwOSwwLjUzMSAtMC4wOTYsMC41NzZjMC4wMjcsMC4wOTEgMC4xMTYsMC4xNiAwLjIwMSwwLjIwM2MwLjA1NCwwLjAyOCAwLjEzMSwwLjAyNSAwLjE4MiwtMC4wMDdjMC4xMDksLTAuMDY4IDAuNDU3LC0wLjgwOSAxLjEyMSwtMi4xMzVjMC4zMTMsLTAuNjI2IDEuNjcxLC0yLjIxNiAyLjY1MSwtMi45NmMwLjkzNSwtMC43MDkgMC45NTYsLTAuNjcyIDEuOTgsLTEuMjQyYzQuODA4LC0yLjY3OSA5LjY5NywtMC45NjkgOS44MjIsLTAuOTM2YzEuNTY5LDAuNDEzIDMuMTE1LDEuMzA3IDMuNDc3LDEuNjIzYzIuNTI1LDIuMjAzIDMuNTk1LDQuNjg2IDQuMzM5LDkuNzhjMC4zODUsMi42MzMgLTAuMDg2LDE2LjA0NiAwLjM4NywxOS42OTdjMC4xNjIsMS4yNSAwLjA5MiwxLjI0OSAwLjIxLDIuNTExYzAuMTksMi4wMzMgMC4xMjcsMi4wMzMgMC4zMDksNC4wNjZjMC4wMDksMC4xMDMgMC4wODEsMC45MDggLTAuMzE4LDEuMjUzYy0wLjMyNCwwLjI4IC0wLjM3NywwLjI1IC02LjU2OSwwLjI1Yy0zLjk2MywtMCAtNC40NjMsMC4wODQgLTQuNzU4LC0wLjUyOGMtMC40MTQsLTAuODU4IDAuMTczLC00LjIxMSAwLjIzOCwtNy4yMzVjMC4wNDQsLTIuMDU3IDAuMTI1LC0yLjA0OCAwLjIxMywtNC4zN2MwLjM3OSwtOS45OSAwLjQyOCwtOS4xNzIgLTAuMTQ0LC0xMS44Yy0wLjk1OSwtNC40MDcgLTMuNDM0LC00Ljc3MyAtNS4yMTQsLTQuODMzWiIgc3R5bGU9ImZpbGw6I2ZkN2Q1OTsiLz48cGF0aCBkPSJNNTAuMzY0LDEwNS40OTJjMC4yMzcsLTAuMjM2IDEuOTkzLC0xLjU2NSAyLjE1OCwtMS42NDljMC4xODksLTAuMDk2IDEuNDI4LC0wLjc4MSAxLjkyNCwtMC44NzdjMC42NywtMC4xMyAwLjk5NywtMC40NjkgNC43MDUsLTAuMzVjMS4yNjcsMC4wNDEgMy4xNjQsMC43MSA0LjAwNSwxLjE0YzAuMDQ1LDAuMDIzIDEuMjE1LDAuODk2IDEuMjQ3LDAuOTJjMC40NjcsMC4zNTIgMS45OTYsMi4yNyAyLjA0MywyLjM0NmMwLjA3NiwwLjEyMyAwLjgzNCwxLjQxIDEuMDM2LDEuNzIzYzAuMTU3LDAuMjQ0IDAuMTUzLDAuMzA3IDAuNDM5LDAuMjc4YzAuMjcxLC0wLjAyNyAwLjE4OCwtMC4xNDIgMC40MDUsLTAuNjI5YzAuMDg5LC0wLjE5OSAwLjYyNSwtMS4xODYgMS4yNDksLTEuODMxYzAuMDc0LC0wLjA3NiAxLjA0NiwtMC45OTggMS40NTMsLTEuMzk5YzAuMDUsLTAuMDQ5IDEuOTcyLC0xLjUyIDQuMDc1LC0yLjEyN2MwLjM3MiwtMC4xMDcgMS4zNzgsLTAuNTIgNC42ODcsLTAuNDI5YzAuOTU4LDAuMDI2IDIuNzc3LDAuNTMyIDMuMTQ0LDAuNjE1YzEuMDUsMC4yMzYgMy45ODcsMS44NiA1LjQ5OCw0LjIzYzAuNjczLDEuMDU2IDEuMTExLDIuOTE5IDEuMzQzLDMuNzAyYzAuMTI5LDAuNDM2IDAuMzg4LDEuNjIzIDAuNjIyLDMuNzUyYzAuMTExLDEuMDA4IC0wLjAyMiwxNC42NDkgMC4yMzUsMTUuOTY5YzAuMzgzLDEuOTcyIDAuNDMxLDYuODU4IDAuNTI1LDguMDk4YzAuMDYxLDAuNzk5IDAuMjI0LDAuNzcyIDAuMjU2LDEuNTY5YzAuMDY1LDEuNTk1IC0wLjM1LDEuNTQ3IC0wLjM3NCwxLjU1OWMtMC40NTEsMC4yMzQgLTAuNDI2LDAuMyAtMC45MjcsMC4zNjdjLTAuNjcsMC4wOSAtMC42NzEsMC4wMzIgLTguNDQ5LDAuMDEyYy0wLjM0NCwtMC4wMDEgLTIuMjQ5LDAuMjgyIC0yLjI3NCwtMS42MjNjLTAuMDMzLC0yLjQ1MyAwLjEyNSwtMy4xNDggMC4xODUsLTMuNDE1YzAuMTQ5LC0wLjY1OCAwLjM3NCwtNS44IDAuMzkzLC02LjU5OGMwLjEwOSwtNC41MzcgMC4wMDcsLTQuNTMxIDAuMDk2LC05LjA2NWMwLjAxNywtMC44NzkgMC4xODQsLTMuNDk3IC0wLjE3MywtNC44MzJjLTAuMTE2LC0wLjQzMyAtMC4xNjQsLTEuMDg4IC0wLjg0NCwtMi40MDljLTAuMDk1LC0wLjE4NSAtMC45NzMsLTEuMjI2IC0xLjc3MSwtMS42MDZjLTEuOTMyLC0wLjkyMSAtMy4zNTYsLTAuNDU3IC0zLjc0OCwtMC4zNDRjLTEuMDY3LDAuMzA3IC0xLjEwOSwwLjQxMiAtMS4zMDcsMC41MzljLTAuMzYyLDAuMjM0IC0xLjY3MiwxLjM3IC0yLjM0MSwzLjI0NmMtMC40NzYsMS4zMzQgLTAuNzc5LDIuMDg2IC0wLjgyNSwzLjgxNmMtMC4wMDUsMC4xODMgLTAuMDgzLDAuMTc3IC0wLjA5NSwzLjc3OWMtMC4wMTQsNC40ODUgMC40NjgsMTIuODQ1IDAuNzIsMTQuMDk4YzAuMTA5LDAuNTQyIDAuMTU3LDAuNTM1IDAuMjA5LDIuNDcyYzAuMDM0LDEuMjY1IC0wLjI0MSwxLjY3MSAtMC43NSwxLjg0OGMtMC4yNjQsMC4wOTIgLTEuODEyLDAuMTA2IC00LjA1MSwwLjEwOWMtNi4wODcsMC4wMDcgLTYuMTI2LDAuMDE1IC02LjU5OSwtMC4zNTNjLTAuOTIyLC0wLjcxOSAtMC41MzIsLTIuMzQ3IC0wLjQ5LC0yLjUwNmMwLjIwMSwtMC43NTggMC4wNjksLTEuNTEgMC4yMjgsLTIuMjE2YzAuNDU3LC0yLjAyOCAwLjU1NCwtMTUuMDA5IDAuNTI0LC0xNi44OTFjLTAuMDM2LC0yLjIzMyAtMC4wNzMsLTAuNzY4IC0wLjIxMywtMS42OGMtMC4xNDcsLTAuOTUzIC0wLjAxNSwtMC44MSAtMC4yMjksLTEuNTljLTAuMzMzLC0xLjIxNSAtMC45NTUsLTIuNzMyIC0xLjI1OCwtMy4xNTVjLTAuOTk2LC0xLjM5MyAtMS43NzksLTEuODc2IC00LjE1NSwtMS43MzdjLTAuMTU1LDAuMDA5IC0wLjYzLDAuMDM3IC0xLjgyLDAuNjU4Yy0wLjI0NiwwLjEyOSAtMS41MjIsMS4wNzkgLTIuMTQxLDIuMjA5Yy0wLjE1MywwLjI4IC0wLjIsMC4yOSAtMC4zNDgsMC44MjhjLTAuMDE1LDAuMDU0IC0wLjIzOCwwLjYxNiAtMC40OSwxLjUwOGMtMC4yMjEsMC43ODEgLTAuNzE3LDQuNjA4IC0wLjY3NSw4LjU4OWMwLjAwOCwwLjc5MiAwLjM3LDkuMTczIDAuNDExLDkuOTYyYzAuMDksMS43MjQgMC4zODEsMi42ODMgMC40MzcsMy40NzhjMC4xNjIsMi4yNjggLTAuMTE1LDIuMzcgLTAuMzgsMi41MTFjLTAuNzU5LDAuNDA0IC0wLjc5NCwwLjM5NSAtNC4zODIsMC4zOWMtNi4xMzUsLTAuMDA4IC02LjE4OSwwLjAzMSAtNi42MjQsLTAuMzE3Yy0wLjE1NiwtMC4xMjUgLTAuNDU3LC0wLjE3NCAtMC4zODksLTEuNjQ1YzAuMDcyLC0xLjU2NCAwLjIyNCwtMi4yNjMgMC41MjksLTMuNTI1YzAuMjY1LC0xLjA5MiAwLjExMywtNS45NTggMC4xMjQsLTExLjE3YzAuMDIsLTkuMTUzIDAuMDM4LC0xMy45NzIgLTAuMTA5LC0xNC43MjNjLTAuMTc5LC0wLjkxMyAtMC40NjksLTIuOTI2IC0wLjUwNSwtNC4wMzdjLTAuMDM1LC0xLjA4MSAwLjA4NCwtMS4zOTIgMC40MTYsLTEuNTU1YzAuMTkzLC0wLjA5NCAwLjE3NiwtMC4xMTMgMC4zNiwtMC4yMTRjMC40MTcsLTAuMjI4IDMuNzgxLC0xLjA4MiA0LjMxMiwtMS4xMzljMC43OTksLTAuMDg1IDIuMzM0LC0wLjU0NSAzLjEzLC0wLjYxNWMyLjM0NCwtMC4yMDYgMi41MzEsMC4zNzEgMi41MzEsMC4zNzFjMCwwIDAuMjUxLDAuMzE1IDAuMjY3LDAuMzQ2YzAuMTUzLDAuMjgzIDAuMTI2LDEuMzAxIDAuMTA0LDIuMThjLTAuMDI0LDAuOTg1IC0wLjU2MywxLjg5NiAtMC43NTMsMy4wMjNjLTAuMDgzLDAuNDk0IC0wLjE0OSwwLjc0MiAwLjM0MiwwLjY1N2MwLjE4MiwtMC4wMzIgMC4yODgsLTAuMzEzIDAuMzAxLC0wLjM0N2MwLjEyNCwtMC4zMyAwLjEzNSwtMC4zMjEgMC4yOTMsLTAuNjRjMC4wNSwtMC4xMDEgMC41ODUsLTEuNDQ4IDEuMDA2LC0yLjA1OGMwLjQxOCwtMC42MDYgMS4yOCwtMS4zNjYgMS41MTgsLTEuNjA0WiIgc3R5bGU9ImZpbGw6I2ZkN2Q1OTsiLz48cGF0aCBkPSJNMjE2Ljg3MSwxMDUuMzczYzAuMDU5LDAuMDQ4IDMuODUxLDIuMjEgNi44MjEsOC44OTNjMC40MDQsMC45MDkgMC4yMzIsMC45NiAwLjU1NiwxLjg5M2MwLjA2MywwLjE4MiAwLjEwMywwLjU2IDAuMTA4LDAuNjFjMC4wOTksMC45NTMgMC41NTQsMi41NCAwLjU5LDQuMDczYzAuMDA1LDAuMjAxIDAuMDc2LDQuMzMyIC0wLjA3NSw1LjAwMWMtMC4yNzcsMS4yMjggLTAuMDUzLDEuMjY2IC0wLjM5NywyLjQ3M2MtMC4yMiwwLjc3MSAtMC4wOSwwLjc5MyAtMC4zMTUsMS41NjJjLTAuMTgyLDAuNjI0IC0wLjEsMC42MzkgLTAuMzE1LDEuMjQ1Yy0wLjE4NiwwLjUyNSAtMC4xNDksMC41MzYgLTAuNDAyLDEuMDNjLTAuMTUxLDAuMjk0IC0wLjQxNSwxLjA5NiAtMC40NDcsMS4xOTJjLTAuMDI5LDAuMDg5IC0wLjA0OSwwLjE4MiAtMC4wOTUsMC4yNjNjLTAuMDI1LDAuMDQ0IC0wLjA4NywwLjA1OCAtMC4xMTMsMC4xMDFjLTAuMDQ4LDAuMDc5IC0wLjA1MSwwLjE4MiAtMC4xMDQsMC4yNThjLTAuMDM1LDAuMDUgLTAuMTExLDAuMDU5IC0wLjE0OCwwLjEwN2MtMC4xNjMsMC4yMTIgLTIuMzIsNy4xNjkgLTEyLjAyMyw5LjA3NmMtMC42MzQsMC4xMjUgLTAuNjM5LDAuMDQ1IC0xLjI2MiwwLjE5NGMtMC4yMzksMC4wNTcgLTMuNDYyLDAuMjE2IC00LjY1NiwtMC4wODhjLTAuOTIzLC0wLjIzNSAtMC45NjIsLTAuMDE0IC0xLjg2OCwtMC4zMTdjLTAuNzksLTAuMjY0IC0xLjY4MiwtMC41MzEgLTEuODY0LC0wLjYyNWMtMC40OTksLTAuMjU5IC0wLjQ5MiwtMC4yNjkgLTEuMjk1LC0wLjUzNWMtMC4zMzksLTAuMTEzIC0wLjMwNCwtMC4xNzQgLTAuNjI5LC0wLjMwMmMtMC4wODYsLTAuMDM0IC0wLjE3OCwtMC4wNTggLTAuMjU4LC0wLjEwNWMtMC4wNDQsLTAuMDI2IC0wLjA2MiwtMC4wODMgLTAuMTA2LC0wLjEwOWMtMC4wNzksLTAuMDQ3IC0wLjE3OSwtMC4wNTUgLTAuMjU1LC0wLjEwN2MtMC4wNSwtMC4wMzQgLTAuMDYzLC0wLjEwNiAtMC4xMTEsLTAuMTQyYy0wLjI0NywtMC4xODYgLTAuNzYzLC0wLjQ0MiAtMC44MzEsLTAuNDc2Yy0wLjI3NywtMC4xMzggLTAuNjE5LC0wLjQ2OSAtMC42OTUsLTAuNTE4Yy0wLjA3NCwtMC4wNDggLTAuMTY5LC0wLjA2IC0wLjIzOSwtMC4xMTRjLTAuMDUzLC0wLjA0MSAtMC4wNjcsLTAuMTE4IC0wLjExOCwtMC4xNjFjLTAuMDY0LC0wLjA1MyAtMC4xNTgsLTAuMDY0IC0wLjIyLC0wLjExOWMtMC4wNTMsLTAuMDQ3IC0wLjA2NSwtMC4xMjkgLTAuMTE3LC0wLjE3N2MtMC4xNzIsLTAuMTYyIC0wLjI2MSwtMC4xNzEgLTAuMzg5LC0wLjI2MWMtNy4zNSwtNS4yMjYgLTYuMzk3LC0xNy41MzUgLTYuMzM1LC0xOC4zMzZjMC4xOTEsLTIuNDU4IDAuNjgxLC00LjE1NCAwLjg1MiwtNS4wMjJjMC4xMDUsLTAuNTM0IDAuMTU0LC0wLjUxMyAwLjcwNCwtMi4xN2MwLjAzLC0wLjA4OSAwLjA1MywtMC4xODIgMC4wOTgsLTAuMjY1YzAuMDIyLC0wLjA0MSAwLjA3NiwtMC4wNiAwLjA5OCwtMC4xMDFjMC4xMzYsLTAuMjQ5IDAuMDcsLTAuMjcxIDAuMjExLC0wLjUyNWMwLjAyMywtMC4wNDIgMC4wNzMsLTAuMDY1IDAuMDk1LC0wLjEwOGMwLjA4NCwtMC4xNjcgMC4xMzMsLTAuMzQ5IDAuMjE4LC0wLjUxNWMwLjAyMywtMC4wNDUgMC4wOCwtMC4wNjUgMC4xMDMsLTAuMTA5YzAuMDg0LC0wLjE2NiAwLjEyNiwtMC4zNSAwLjIxMiwtMC41MTVjMC4wMjQsLTAuMDQ1IDAuMDg2LC0wLjA2MSAwLjExMiwtMC4xMDVjMC4wNDgsLTAuMDc5IDAuMDU0LC0wLjE4IDAuMTA1LC0wLjI1NmMwLjAzMywtMC4wNDkgMC4xMDUsLTAuMDYxIDAuMTQsLTAuMTA4YzAuMDI0LC0wLjAzMyAwLjEzLC0wLjQ1MiAwLjg3MywtMS40MDdjMi42MTcsLTMuMzY0IDYuNDksLTUuNDQ0IDcuMSwtNS42ODljMC40NjMsLTAuMTg2IDAuNDM2LC0wLjI0OSAwLjkyMSwtMC4zNzNjMC40ODgsLTAuMTI0IDAuNDU2LC0wLjIxNCAwLjk0NSwtMC4zMDZjMC4xOTcsLTAuMDM3IDAuNCwtMC4wNDQgMC41OTQsLTAuMDkzYzAuMTI2LC0wLjAzMiAwLjIzNywtMC4xMDggMC4zNiwtMC4xNDhjMC4wNDcsLTAuMDE2IDAuMjE4LC0wLjA3MiAwLjYxMSwtMC4xMDZjMC42NDQsLTAuMDU2IDAuNjQsLTAuMDU3IDEuMjY1LC0wLjIwNGMwLjcxNiwtMC4xNjkgMy44OTgsLTAuMTk1IDQuNjgxLC0wLjAwNGMwLjc3OCwwLjE5IDAuNzg5LDAuMDk4IDEuNTc4LDAuMjQxYzAuMTAyLDAuMDE4IDAuMDg2LDAuMDc3IDEuMjUzLDAuMzA3YzAuOTA4LDAuMTc5IDIuMDY0LDAuNzkxIDIuMjM0LDAuODgxYzAuMjQ5LDAuMTMyIDAuMjY2LDAuMDgzIDAuNTE0LDAuMjI1YzAuODEsMC40NjEgMS40OTgsMC43NjggMS42MDksMC44NDdjMC4wNTIsMC4wMzggMC4wNywwLjExMSAwLjEyLDAuMTUyWm0tMi4xNzksMTkuNzIyYy0wLjEwMywtMy44MjEgMC4xNDIsLTEzLjk5MSAtNy45MzMsLTEzLjk1M2MtNS41MjQsMC4wMjYgLTcuNjU0LDYuMjcyIC03LjU0MywxNC4wOThjMC4wNDMsMy4wNDQgMS4yNDIsMTAuMDQxIDguMTU4LDkuODk0YzQuOTM1LC0wLjEwNSA3LjQyLC02LjIxNyA3LjMxOCwtMTAuMDM4WiIgc3R5bGU9ImZpbGw6I2ZkN2Q1OTsiLz48cGF0aCBkPSJNMTg0LjAzNCwxMTAuODZjMC4wMDcsMC4zIC0wLjA4MywwLjQwMyAtMC4zNTgsMC44MjZjLTAuMzksMC42MDIgLTEuNzc0LDAuNTE5IC0zLjUzNiwtMC4xNzdjLTUuMzA1LC0yLjA5NCAtMTAuODI2LC0xLjMxNyAtMTAuNzE2LDIuMTQ2YzAuMTI4LDQuMDAzIDMuNjEyLDMuNzIzIDUuNzQ3LDQuNTg3YzIuNTM3LDEuMDI3IDguNTE0LDMuNTI3IDEwLjcxNiw5Ljc2NWMwLjg1MSwyLjQxMiAwLjYxMiw0LjM0NyAwLjU2NSw0LjcyNWMtMC4xMTEsMC44OTQgLTAuNjA1LDYuODQgLTcuNTk3LDkuNTQ0Yy0xLjAzMiwwLjM5OSAtMi44NDQsMC43MjEgLTMuMzY0LDAuODI0Yy0wLjU3NCwwLjExNCAtNC45ODIsMC40ODggLTguMTY0LC0wLjI2MmMtMC45MiwtMC4yMTcgLTEuMzQ4LC0wLjE2NiAtMi41MDEsLTAuNjc2Yy0xLjEzNSwtMC41MDIgLTIuODY5LC0xLjYzNCAtMy45ODgsLTIuMzk3Yy0wLjY4NywtMC40NTcgLTEuMTIzLC0xLjIwOCAtMS4xOCwtMi4wMzFjLTAuMDcyLC0xLjA1IC0wLjE1MiwtMi42MTkgLTAuMTQ4LC00LjJjMC4wMDEsLTAuNTI0IDAuMDk5LC0xLjcwNyAxLjA4MSwtMS43NDNjMC44MDksLTAuMDMgMS4xMDYsLTAuMTY1IDIuNDA1LDAuNzY3YzAuMDk3LDAuMDcgMS4xNjcsMC44MjggMy4zNTcsMS44NDljNS4wMTMsMi4zMzcgOC44NjQsMi4wMyAxMC4xNjYsLTAuMjI5YzAuMjI1LC0wLjM5IDAuMzU0LC0wLjY2MyAwLjM3OSwtMS41OTJjMC4wODgsLTMuMjI0IC0zLjA0NywtNC4zODIgLTQuNTc2LC01LjEwOWMtMS4yNjYsLTAuNjAyIC0xLjIzOSwtMC42NDYgLTIuNTE2LC0xLjIyN2MtMC4yMjQsLTAuMTAyIC00LjAxNSwtMS45NSAtNS4yMzcsLTIuOTUxYy0xLjA0LC0wLjg1MSAtMS40NTcsLTEuMTg3IC0yLjYxNiwtMi43MWMtMS41MTYsLTEuOTkyIC0yLjMxNiwtNS4zODIgLTEuODc5LC03Ljg4N2MwLjA3NiwtMC40MzYgMC4yNjIsLTMuODE1IDMuNzI5LC02Ljk5M2MxLjE1NiwtMS4wNiAxLjI1OCwtMC45MTIgMi42NDQsLTEuNjY5YzAuODksLTAuNDg3IDQuNjUxLC0xLjU5NyA3Ljg0MSwtMS4yNTdjMy4wMzcsMC4zMjMgNS40NzksMS4wMTggNy4wODYsMS41OTFjMS4zMjYsMC40ODMgMi4yNTEsMS42OTIgMi4zNywzLjA5OGMwLjEzMywxLjMxMSAwLjI3NCwyLjgyNSAwLjI4OCwzLjM4N1oiIHN0eWxlPSJmaWxsOiNmZDdkNTk7Ii8+PHBhdGggZD0iTTE0NC43MDksMTQwLjg2NmMtMC4wMDIsMC4xMDEgMC4wMTIsMC4yMDIgMC4wMSwwLjMwM2MtMC4wMzMsMS40NDUgLTIuMTk1LDEuMTkgLTIuNjk5LDEuMTkxYy0wLjg3NywwLjAwMiAtNy42MjksMC4wMDUgLTcuODE0LC0wLjAwOWMtMS4yNCwtMC4wOTYgLTEuMjEsLTEuMzM4IC0xLjIwNiwtMS40OTFjMC4wNDIsLTEuNzI5IDAuMjg4LC0xLjcwNSAwLjM2NiwtMy40MzNjMC4yMjIsLTQuODc3IDAuNTE4LC04LjcyMiAwLjU0NiwtMTAuNjQxYzAuMTU0LC0xMC44ODkgMC4wNzIsLTEwLjg4OSAtMC40NDYsLTE2LjU2M2MtMC4xMDEsLTEuMTAyIDAuMDMyLC0xLjIzNiAtMC4zMjIsLTIuODAzYy0wLjQ1NSwtMi4wMTggMC40NjYsLTEuOTg2IDMuODc1LC0yLjgyNWMwLjkyMywtMC4yMjcgMC45MzYsLTAuMTUxIDEuNTU5LC0wLjMyOGMwLjM1NywtMC4xMDIgMy43NDgsLTEuMDY4IDQuNjM2LC0wLjI1MWMxLjAzMywwLjk1MSAwLjY1OCwxLjg2NiAwLjU5Nyw1LjI1N2MtMC4wMzMsMS44MjcgLTAuMjA2LDIuNTkyIC0wLjI1NiwyLjgxMmMtMC4wMTEsMC4wNDkgLTAuMTM0LDAuNTkxIC0wLjEzLDAuNjEzYzAuMDA3LDAuMDQ1IDAuMDgzLDAuNTI5IDAuNDcxLDAuMzI3YzAuMTU4LC0wLjA4MiAwLjcxNCwtMi41NTUgMi43MjgsLTUuMzczYzAuNzcsLTEuMDc4IDAuNzQ3LC0xLjI2NyAyLjQ3NSwtMi44N2MwLjE0MywtMC4xMzMgMC44NDQsLTAuNzgzIDIuMDU4LC0xLjMxNGMxLjUxNSwtMC42NjMgMi41MjIsLTAuNjgxIDIuNzQ5LC0wLjY4NmMxLjI5LC0wLjAyMyA0LjE4NywwLjQwMyA0LjE2OCwyLjQyNmMtMC4wMDMsMC4zNDYgLTAuMTQxLDEuMTQ1IC0wLjE3OCwxLjU1NGMtMC4xMzUsMS41MzEgLTAuNTY5LDMuMzk1IC0wLjYxMiwzLjc1Yy0wLjI2NSwyLjE4MyAtMC40MDIsMi44OTUgLTIuMDgxLDIuNDI4Yy0xLjI2OSwtMC4zNTIgLTYuMTA3LDAuNjEyIC04LjcxMiw1LjQ1M2MtMS43MzMsMy4yMjIgLTIuMjksNi44OSAtMi40NjEsOC4yODVjLTAuNDM1LDMuNTUxIC0wLjE2NCw5LjIzNiAwLjY3OSwxNC4xODRaIiBzdHlsZT0iZmlsbDojZmQ3ZDU5OyIvPjxwYXRoIGQ9Ik0xMy4zODQsMTMxLjg5N2MwLjMyNCwwLjMyMSAwLjMwNCwwLjMzOCAwLjY2OSwwLjYxNGMwLjYwNiwwLjQ1OCAxLjIyMywwLjg3OSAyLjIxLDEuMjU3YzEuMzk1LDAuNTM0IDEuOTE0LDAuNDY4IDIuMjI1LDAuNTQzYzAuODI5LDAuMiAwLjgzMywwLjE4NiAyLjgxOSwwLjE3MmMwLjc5NCwtMC4wMDYgMi42MjIsLTAuMjk3IDIuNzc0LC0wLjM0M2MwLjAwMywtMC4wMDEgMS4wMDUsLTAuMzI0IDEuNTg1LC0wLjU0OWMwLjgsLTAuMzExIDIuNDYyLC0xLjA5MSAyLjg5NCwtMS4zOTRjMC41MjcsLTAuMzcgMC41MjcsLTAuMzY2IDAuNTczLC0wLjM5N2MwLjk5MiwtMC42NjcgMS44LC0wLjk1NCAyLjE3MywtMC45MDZjMC4yNTMsMC4wMzIgMC42MjEsMC4xODQgMC43NjIsMC40NDFjMC4yMjYsMC40MTEgMC42NjcsMS4wNDMgMC41OTIsMi4wMjZjLTAuMDIzLDAuMjk5IC0wLjA0NiwwLjU3MiAtMC4yMDcsMC44NzZjLTAuMDIyLDAuMDQxIC0xLjM2NiwyLjU4MyAtMi4xODMsMy4zODhjLTAuMjExLDAuMjA3IC0xLjg4NSwxLjg1NSAtMi4wNSwyLjAxNmMtMC43NDUsMC43MyAtMS42MDUsMS4zMzUgLTEuNjk0LDEuNDAzYy0wLjAzMywwLjAyNSAtMC45MDUsMC40NzMgLTEuMjA3LDAuNjVjLTAuMDk2LDAuMDU2IC0wLjEwNCwwLjAzNyAtMS4yNDgsMC42MTljLTAuMTY3LDAuMDg1IC0xLjE2MiwwLjQwMyAtMS44MDUsMC41MzdjLTAuMzgyLDAuMDggLTIuMDMsMC41NCAtMi44MzMsMC41NjFjLTMuNjI5LDAuMDk1IC01LjE1LC0wLjIyNiAtNS45NDksLTAuMjgyYy0wLjkzOCwtMC4wNjYgLTMuODgyLC0xLjA2NyAtNC4wMDMsLTEuMTM0Yy0wLjA0MywtMC4wMjQgLTMuMjExLC0yLjAxNiAtMy4zOTMsLTIuMTg1Yy0wLjAyOCwtMC4wMjYgLTEuNDMsLTEuMjg5IC0yLjAxMiwtMi4wNDhjLTAuNjU5LC0wLjg1OSAtMS44NjgsLTIuOTM3IC0xLjk0MSwtMy4xMDljLTEuNjI2LC0zLjgxNyAtMS44NDgsLTUuMTY5IC0yLjA3MSwtNy44NzNjLTAuMjI3LC0yLjc2NSAwLjE4NywtNi4yMTggMC43MDcsLTguNzE3YzAuNDU3LC0yLjE5NiAyLjM3NCwtNi4xOTcgMi40MTMsLTYuMjhjMC4xMiwtMC4yNTQgMy45NDUsLTkuNDQzIDE1LjMwNSwtOS4zNjdjMS41OTgsMC4wMTEgMTIuMzg3LDAuMjE1IDE0LjA1OCwxMy4xMThjMC4yNjYsMi4wNTMgMC4zNTUsNC44NSAtMC4wODcsNi4yMWMtMC4wNTEsMC4xNTcgLTAuNDg4LDEuNSAtMC44NzcsMS44NjZjLTAuMDYxLDAuMDU3IC0wLjE1MSwwLjA3NSAtMC4yMTgsMC4xMjZjLTAuMjU1LDAuMTkzIC0wLjI0OCwwLjI4NSAtMS4yOTksMC4yODljLTYuMjU5LDAuMDI0IC0xMC45MTYsMC4wMiAtMTQuMDksMC4wMmMtMy40MywtMCAtNS4xMjcsMC4wMDUgLTUuMjM5LDAuMDU2Yy0wLjA1MiwwLjAyNCAtMC4wNzQsMC4wODcgLTAuMTE1LDAuMTI3Yy0wLjExMSwwLjEwOSAtMC4yNCwwLjA5NyAtMC4xNjIsMS4yOThjMC4wNTEsMC43OTYgMC45NTksMy4zNjMgMS4wNDIsMy41MDVjMC4xNDYsMC4yNDkgMC44ODUsMS42OTIgMS4yNjYsMi4xOTZjMC4yNzYsMC4zNjQgMC4yOTIsMC4zNDMgMC42MTYsMC42NjlabTQuNTIxLC0xMy42OTdjMC44MzksLTAuMDA1IDEuNzMzLC0wLjAxIDMuNDA1LC0wLjAxNmMxLjU4OSwtMC4wMDUgMi4yMTcsLTAuMjIzIDIuMDYyLC0yLjk2MmMtMC4xNzIsLTMuMDE5IC0xLjkzNCwtNS41NjMgLTQuNTgzLC02LjA1NGMtMC4zMSwtMC4wNTcgLTMuNTYzLC0wLjYwMSAtNi4yMTIsMy4wNDJjLTAuNDc0LDAuNjUxIC0xLjkxMywyLjMyOCAtMi4xMTYsNS4xOWMtMC4wMDQsMC4wNTEgMC4wMiwwLjM2MSAwLjI1NCwwLjU4NWMwLjI4OSwwLjI3NiAwLjk3NywwLjIyNyAzLjQwMSwwLjIyNWMxLjkyOCwtMC4wMDEgMi44MjIsLTAuMDA1IDMuNzg5LC0wLjAxMVoiIHN0eWxlPSJmaWxsOiNmZDdkNTk7Ii8+PHBhdGggZD0iTTEwOC41MzUsMTMxLjg5N2MwLjMyNCwwLjMyMSAwLjMwNCwwLjMzOCAwLjY2OSwwLjYxNGMwLjYwNiwwLjQ1OCAxLjIyMywwLjg3OSAyLjIxLDEuMjU3YzEuMzk1LDAuNTM0IDEuOTE0LDAuNDY4IDIuMjI1LDAuNTQzYzAuODI5LDAuMiAwLjgzMywwLjE4NiAyLjgxOSwwLjE3MmMwLjc5NCwtMC4wMDYgMi42MjIsLTAuMjk3IDIuNzc0LC0wLjM0M2MwLjAwMywtMC4wMDEgMS4wMDUsLTAuMzI0IDEuNTg1LC0wLjU0OWMwLjgsLTAuMzExIDIuNDYyLC0xLjA5MSAyLjg5NCwtMS4zOTRjMC41MjcsLTAuMzcgMC41MjcsLTAuMzY2IDAuNTczLC0wLjM5N2MwLjk5MiwtMC42NjcgMS44LC0wLjk1NCAyLjE3MywtMC45MDZjMC4yNTMsMC4wMzIgMC42MjEsMC4xODQgMC43NjIsMC40NDFjMC4yMjYsMC40MTEgMC42NjcsMS4wNDMgMC41OTIsMi4wMjZjLTAuMDIzLDAuMjk5IC0wLjA0NiwwLjU3MiAtMC4yMDcsMC44NzZjLTAuMDIyLDAuMDQxIC0xLjM2NiwyLjU4MyAtMi4xODMsMy4zODhjLTAuMjExLDAuMjA3IC0xLjg4NSwxLjg1NSAtMi4wNSwyLjAxNmMtMC43NDUsMC43MyAtMS42MDUsMS4zMzUgLTEuNjk0LDEuNDAzYy0wLjAzMywwLjAyNSAtMC45MDUsMC40NzMgLTEuMjA3LDAuNjVjLTAuMDk2LDAuMDU2IC0wLjEwNCwwLjAzNyAtMS4yNDgsMC42MTljLTAuMTY3LDAuMDg1IC0xLjE2MiwwLjQwMyAtMS44MDUsMC41MzdjLTAuMzgyLDAuMDggLTIuMDMsMC41NCAtMi44MzMsMC41NjFjLTMuNjI5LDAuMDk1IC01LjE1LC0wLjIyNiAtNS45NDksLTAuMjgyYy0wLjkzOCwtMC4wNjYgLTMuODgyLC0xLjA2NyAtNC4wMDMsLTEuMTM0Yy0wLjA0MywtMC4wMjQgLTMuMjExLC0yLjAxNiAtMy4zOTMsLTIuMTg1Yy0wLjAyOCwtMC4wMjYgLTEuNDMsLTEuMjg5IC0yLjAxMiwtMi4wNDhjLTAuNjU5LC0wLjg1OSAtMS44NjgsLTIuOTM3IC0xLjk0MSwtMy4xMDljLTEuNjI2LC0zLjgxNyAtMS44NDgsLTUuMTY5IC0yLjA3MSwtNy44NzNjLTAuMjI3LC0yLjc2NSAwLjE4NywtNi4yMTggMC43MDcsLTguNzE3YzAuNDU3LC0yLjE5NiAyLjM3NCwtNi4xOTcgMi40MTMsLTYuMjhjMC4xMiwtMC4yNTQgMy45NDUsLTkuNDQzIDE1LjMwNSwtOS4zNjdjMS41OTgsMC4wMTEgMTIuMzg3LDAuMjE1IDE0LjA1OCwxMy4xMThjMC4yNjYsMi4wNTMgMC4zNTUsNC44NSAtMC4wODcsNi4yMWMtMC4wNTEsMC4xNTcgLTAuNDg4LDEuNSAtMC44NzcsMS44NjZjLTAuMDYxLDAuMDU3IC0wLjE1MSwwLjA3NSAtMC4yMTgsMC4xMjZjLTAuMjU1LDAuMTkzIC0wLjI0OCwwLjI4NSAtMS4yOTksMC4yODljLTYuMjU5LDAuMDI0IC0xMC45MTYsMC4wMiAtMTQuMDksMC4wMmMtMy40MywtMCAtNS4xMjcsMC4wMDUgLTUuMjM5LDAuMDU2Yy0wLjA1MiwwLjAyNCAtMC4wNzQsMC4wODcgLTAuMTE1LDAuMTI3Yy0wLjExMSwwLjEwOSAtMC4yNCwwLjA5NyAtMC4xNjIsMS4yOThjMC4wNTEsMC43OTYgMC45NTksMy4zNjMgMS4wNDIsMy41MDVjMC4xNDYsMC4yNDkgMC44ODUsMS42OTIgMS4yNjYsMi4xOTZjMC4yNzYsMC4zNjQgMC4yOTIsMC4zNDMgMC42MTYsMC42NjlabTQuNTIxLC0xMy42OTdjMC44MzksLTAuMDA1IDEuNzMzLC0wLjAxIDMuNDA1LC0wLjAxNmMxLjU4OSwtMC4wMDUgMi4yMTcsLTAuMjIzIDIuMDYyLC0yLjk2MmMtMC4xNzIsLTMuMDE5IC0xLjkzNCwtNS41NjMgLTQuNTgzLC02LjA1NGMtMC4zMSwtMC4wNTcgLTMuNTYzLC0wLjYwMSAtNi4yMTIsMy4wNDJjLTAuNDc0LDAuNjUxIC0xLjkxMywyLjMyOCAtMi4xMTYsNS4xOWMtMC4wMDQsMC4wNTEgMC4wMiwwLjM2MSAwLjI1NCwwLjU4NWMwLjI4OSwwLjI3NiAwLjk3NywwLjIyNyAzLjQwMSwwLjIyNWMxLjkyOCwtMC4wMDEgMi44MjIsLTAuMDA1IDMuNzg5LC0wLjAxMVoiIHN0eWxlPSJmaWxsOiNmZDdkNTk7Ii8+PHBhdGggZD0iTTE2MS40NzQsNDkuMDU1Yy0wLjU0OSwtMC43MzMgLTEsLTEuNDU4IC0xLjQ2OSwtMS43M2MtMC4yNywtMC4xNTcgLTIuMDcyLC0yLjUwNSAtMi4yMzUsLTMuMzk1Yy0xLjEzNSwtNi4yMTUgNy4zNjIsLTEuNzk2IDEzLjM1OSwwLjU2NWM0LjQ1NywxLjc1NSAzLjY2NCwzLjYwMiAwLjk3Niw3LjM5NmMtNC43ODgsNi43NTkgLTYuNDI2LDkuMDI2IC04LjQ4OSw3LjQ1OWMtMS42MiwtMS4yMyAtMC41NTIsLTQuMTkxIC0wLjU1OCwtNC44NjdjLTAuMDE5LC0yLjIyMyAtMC41MzgsLTQuMDM0IC0xLjU4NCwtNS40MjlaIiBzdHlsZT0iZmlsbDojOTI0ZmFjOyIvPjxwYXRoIGQ9Ik0xNjEuOTAyLDU0LjIyNmMtMC4xNDQsMC40MTggLTAuNTMzLDAuODkxIC0wLjg5MSwxLjEwOWMtNS44MjcsMy41NTYgLTYuMDM1LDUuMjI0IC0xMi43NTEsMTIuMTA2Yy0xMi4zMDIsMTIuNjA2IC0yMy40MzgsMTMuNjUgLTI0LjU3LDEzLjkwOWMtMy4wNzQsMC43MDMgLTguOTA4LDAuMiAtMTAuMzU2LC0wLjIyYy03LjU2NSwtMi4xOTEgLTExLjM1NCwtNS4yNSAtMTQuNTQ2LC0xMS43NWMtMC42ODUsLTEuMzk1IC0xLjU0OCwtNS41ODkgLTEuNTUsLTUuNjA4Yy0wLjgyLC01Ljk3NSAtMC4xMDgsLTguNDY2IC0wLjA4NSwtOC41OTljMC4yOTksLTEuNjkxIC0wLjk4NSwtMC43NjIgLTMuMTk5LDAuNDU4Yy00Ljg1NCwyLjY3MyAtNC40MjYsLTIuNzk4IC0zLjk4LC0zLjc4NWMxLjkzNiwtNC4yNzkgNS4zMzMsLTUuNDQ5IDguMjA2LC03LjA5NGMxLjA3NiwtMC42MTYgMC41OSwtMi43OTMgMy4xMjksLTkuNjMzYzMuMDYyLC04LjI0OCA1LjMxLC0xMS4xMDMgNy4zNiwtMTQuMjcyYzMuMjY5LC01LjA1NCA3Ljk4OSwtMTAuMDk1IDExLjcwMywtMTMuMTAxYzExLjY3OCwtOS40NTEgMjMuMjQ1LC04LjU2NiAyNy41MTgsLTYuMDk1YzguMTg1LDQuNzM0IDUuMDY2LDEzLjk2OCA0LjUzMiwxNS43NjVjLTAuNjMsMS42IC0wLjgyMiwyLjc1NSAtMy4xNzIsNi4zNDZjLTAuNzc5LDEuMTkgLTQuNzQyLDcuMjQ1IC0xMi40MzMsMTIuNzAyYy05LjM0NSw2LjYzMSAtMjAuMzgsMTAuNjI5IC0yMy42NTcsMTEuODk4Yy0xLjQ2LDAuNTY1IC0xLjE2OSwwLjkwNSAtMS4zMDQsMS43MDVjLTAuOTMxLDUuNTQ0IC0wLjY3NSw4LjE5MiAwLjcwNCwxMS45NzhjMi4xODQsNS45OTggMTIuNjI4LDEwLjUyNyAyNS4zNTQsMS43MjJjNi45MTIsLTQuNzgyIDExLjc2NiwtMTEuNTcxIDE5LjU1NCwtMTUuMjM0YzAuMjAxLC0wLjA5NSAwLjg0NSwtMC40MDMgMS40MywtMC4yNzVjMC40OTUsMC4xMDkgMC43ODgsMC4zMSAwLjkyLDAuNDhjMC40MzUsMC41NTkgMC43NTMsMC44ODEgMS4wODEsMS40OThjMC41NDgsMS4wMzMgMC44NjEsMS43NTkgMS4wMzYsMi40MWMwLjE3OSwwLjY2NiAwLjExOSwxLjEzMSAtMC4wMzYsMS41OFptLTQ2LjUzNSwtMTguMjI1Yy0wLjk5NSwyLjczNCAtMS4wNDEsMi44OTUgLTAuOCwyLjkyOWMwLjI2MSwwLjAzNyAxMC4xNjUsLTMuNzg1IDE1LjQwNSwtNy4zODNjMC4yNSwtMC4xNzEgOC4wNDUsLTQuNzE5IDExLjcxMSwtMTIuNTVjMy4yNDMsLTYuOTI4IDAuNjczLC0xMC4yMzYgLTEuNDc5LC0xMC41MjhjLTEzLjE4MiwtMS43ODcgLTIzLjY4OSwyMy4zMjcgLTI0LjgzNywyNy41MzFaIiBzdHlsZT0iZmlsbDojZmQ3ZDU5OyIvPjwvc3ZnPg==" alt="Emerson">
    </div>
    <div class="login-heading">Podcast Editor</div>
    <div class="login-sub">Acesso restrito</div>

    <div class="login-field">
      <label class="login-label">Usuário</label>
      <input class="login-input" type="text" id="loginUser" placeholder="seu usuário" autocomplete="username">
    </div>
    <div class="login-field">
      <label class="login-label">Senha</label>
      <input class="login-input" type="password" id="loginPass" placeholder="••••••••" autocomplete="current-password">
    </div>

    <div class="login-error" id="loginError"></div>
    <button class="login-btn" onclick="doLogin()">Entrar</button>

    <hr class="login-divider">
    <div class="login-footer">
      Emerson Healthtech · Plataforma interna
    </div>
  </div>
</div>

<!-- ══ APPROVAL SCREEN ══════════════════════════════════════ -->
<div id="approval-screen">
  <div class="approval-box">
    <div class="approval-icon">🎙</div>
    <div class="approval-heading">Aguardando autorização</div>
    <div class="approval-sub">
      Um administrador precisa aprovar seu acesso.<br>
      Aguarde a confirmação ou entre em contato.
    </div>

    <div class="approval-user-badge">
      <div class="badge-avatar" id="approvalAvatar">—</div>
      <div class="badge-info">
        <div class="badge-name" id="approvalName">—</div>
        <div class="badge-role" id="approvalRole">Editor</div>
      </div>
    </div>

    <!-- Botões visíveis apenas para admin -->
    <div class="approval-actions" id="approvalActions" style="display:none">
      <button class="btn-approve" onclick="approveAccess()">✓ Autorizar acesso</button>
      <button class="btn-deny"    onclick="denyAccess()">Negar</button>
    </div>

    <div class="approval-pending-msg" id="approvalPending">
      <div class="pending-dot"></div>
      Aguardando aprovação do administrador
    </div>
  </div>
</div>

<!-- ══ MAIN APP ══════════════════════════════════════════════ -->
<div id="main-app" style="display:none">

<div id="topbar">
  <div class="topbar-logo">
    <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj48c3ZnIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIHZpZXdCb3g9IjAgMCAyNjQgMTQ0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHhtbDpzcGFjZT0icHJlc2VydmUiIHhtbG5zOnNlcmlmPSJodHRwOi8vd3d3LnNlcmlmLmNvbS8iIHN0eWxlPSJmaWxsLXJ1bGU6ZXZlbm9kZDtjbGlwLXJ1bGU6ZXZlbm9kZDtzdHJva2UtbGluZWpvaW46cm91bmQ7c3Ryb2tlLW1pdGVybGltaXQ6MjsiPjxwYXRoIGQ9Ik0yNDcuMTIyLDExMy41OTZjLTAuNjE5LDAuMDExIC0wLjkzMywtMC4wMTUgLTEuNTU2LDAuMTUyYy0xLjYxNCwwLjQzNCAtMi42NjUsMC44NzQgLTQuMDI4LDIuODE0Yy0xLjExLDEuNTggLTEuNjQyLDQuNjAzIC0xLjcwMiw1LjI3NGMtMC40MTQsNC42MyAtMC4wNjIsMTEuNDAyIC0wLjA1OSwxMS41MmMwLjA1NCwyLjQwOSAwLjE4NiwyLjM5NiAwLjMxMSwzLjc2M2MwLjM5MSw0LjI4IDAuNDA3LDQuMzA2IDAuMjA5LDQuNzI2Yy0wLjI3NCwwLjU4IC0xLjEwMSwwLjUxOSAtNC4xNDgsMC41MTljLTYuMzczLDAuMDAxIC02LjQzOSwwLjA0IC02LjgzOSwtMC4zMDVjLTAuODIzLC0wLjcwNyAtMC40ODMsLTEuNzIxIC0wLjM3NywtMi40NTZjMC4wNDQsLTAuMzAxIDAuNjc1LC0xMC4zNjIgMC43MjYsLTE1LjYzMmMwLjA3NCwtNy43NjQgLTAuMDM2LC05LjIwOCAtMC4wOTcsLTEwLjAwNmMtMC4yNDIsLTMuMTcyIC0wLjIyNywtMy4xNyAtMC4yNTUsLTMuNDQ1Yy0wLjAyNiwtMC4yNSAtMC4yODQsLTIuNzExIC0wLjM1NiwtMy4xMjNjLTAuMjQyLC0xLjM2OCAtMC4zMTUsLTEuODY2IDEuOTA0LC0yLjQwMWMxLjQwNywtMC4zMzkgMS4zOCwtMC40NDEgMi44MDcsLTAuNjU3YzAuNTMzLC0wLjA4MSAxLjA1NSwtMC4zNjIgMi44MTYsLTAuNTk0YzAuOTc0LC0wLjEyOCAzLjIzMSwtMS4wODcgMy4xODEsMS40NjVjLTAuMDQ1LDIuMjM4IC0wLjU0LDQuMTcgLTAuNTY2LDQuNDdjLTAuMDQ2LDAuNTM4IC0wLjEwOSwwLjUzMSAtMC4wOTYsMC41NzZjMC4wMjcsMC4wOTEgMC4xMTYsMC4xNiAwLjIwMSwwLjIwM2MwLjA1NCwwLjAyOCAwLjEzMSwwLjAyNSAwLjE4MiwtMC4wMDdjMC4xMDksLTAuMDY4IDAuNDU3LC0wLjgwOSAxLjEyMSwtMi4xMzVjMC4zMTMsLTAuNjI2IDEuNjcxLC0yLjIxNiAyLjY1MSwtMi45NmMwLjkzNSwtMC43MDkgMC45NTYsLTAuNjcyIDEuOTgsLTEuMjQyYzQuODA4LC0yLjY3OSA5LjY5NywtMC45NjkgOS44MjIsLTAuOTM2YzEuNTY5LDAuNDEzIDMuMTE1LDEuMzA3IDMuNDc3LDEuNjIzYzIuNTI1LDIuMjAzIDMuNTk1LDQuNjg2IDQuMzM5LDkuNzhjMC4zODUsMi42MzMgLTAuMDg2LDE2LjA0NiAwLjM4NywxOS42OTdjMC4xNjIsMS4yNSAwLjA5MiwxLjI0OSAwLjIxLDIuNTExYzAuMTksMi4wMzMgMC4xMjcsMi4wMzMgMC4zMDksNC4wNjZjMC4wMDksMC4xMDMgMC4wODEsMC45MDggLTAuMzE4LDEuMjUzYy0wLjMyNCwwLjI4IC0wLjM3NywwLjI1IC02LjU2OSwwLjI1Yy0zLjk2MywtMCAtNC40NjMsMC4wODQgLTQuNzU4LC0wLjUyOGMtMC40MTQsLTAuODU4IDAuMTczLC00LjIxMSAwLjIzOCwtNy4yMzVjMC4wNDQsLTIuMDU3IDAuMTI1LC0yLjA0OCAwLjIxMywtNC4zN2MwLjM3OSwtOS45OSAwLjQyOCwtOS4xNzIgLTAuMTQ0LC0xMS44Yy0wLjk1OSwtNC40MDcgLTMuNDM0LC00Ljc3MyAtNS4yMTQsLTQuODMzWiIgc3R5bGU9ImZpbGw6I2ZkN2Q1OTsiLz48cGF0aCBkPSJNNTAuMzY0LDEwNS40OTJjMC4yMzcsLTAuMjM2IDEuOTkzLC0xLjU2NSAyLjE1OCwtMS42NDljMC4xODksLTAuMDk2IDEuNDI4LC0wLjc4MSAxLjkyNCwtMC44NzdjMC42NywtMC4xMyAwLjk5NywtMC40NjkgNC43MDUsLTAuMzVjMS4yNjcsMC4wNDEgMy4xNjQsMC43MSA0LjAwNSwxLjE0YzAuMDQ1LDAuMDIzIDEuMjE1LDAuODk2IDEuMjQ3LDAuOTJjMC40NjcsMC4zNTIgMS45OTYsMi4yNyAyLjA0MywyLjM0NmMwLjA3NiwwLjEyMyAwLjgzNCwxLjQxIDEuMDM2LDEuNzIzYzAuMTU3LDAuMjQ0IDAuMTUzLDAuMzA3IDAuNDM5LDAuMjc4YzAuMjcxLC0wLjAyNyAwLjE4OCwtMC4xNDIgMC40MDUsLTAuNjI5YzAuMDg5LC0wLjE5OSAwLjYyNSwtMS4xODYgMS4yNDksLTEuODMxYzAuMDc0LC0wLjA3NiAxLjA0NiwtMC45OTggMS40NTMsLTEuMzk5YzAuMDUsLTAuMDQ5IDEuOTcyLC0xLjUyIDQuMDc1LC0yLjEyN2MwLjM3MiwtMC4xMDcgMS4zNzgsLTAuNTIgNC42ODcsLTAuNDI5YzAuOTU4LDAuMDI2IDIuNzc3LDAuNTMyIDMuMTQ0LDAuNjE1YzEuMDUsMC4yMzYgMy45ODcsMS44NiA1LjQ5OCw0LjIzYzAuNjczLDEuMDU2IDEuMTExLDIuOTE5IDEuMzQzLDMuNzAyYzAuMTI5LDAuNDM2IDAuMzg4LDEuNjIzIDAuNjIyLDMuNzUyYzAuMTExLDEuMDA4IC0wLjAyMiwxNC42NDkgMC4yMzUsMTUuOTY5YzAuMzgzLDEuOTcyIDAuNDMxLDYuODU4IDAuNTI1LDguMDk4YzAuMDYxLDAuNzk5IDAuMjI0LDAuNzcyIDAuMjU2LDEuNTY5YzAuMDY1LDEuNTk1IC0wLjM1LDEuNTQ3IC0wLjM3NCwxLjU1OWMtMC40NTEsMC4yMzQgLTAuNDI2LDAuMyAtMC45MjcsMC4zNjdjLTAuNjcsMC4wOSAtMC42NzEsMC4wMzIgLTguNDQ5LDAuMDEyYy0wLjM0NCwtMC4wMDEgLTIuMjQ5LDAuMjgyIC0yLjI3NCwtMS42MjNjLTAuMDMzLC0yLjQ1MyAwLjEyNSwtMy4xNDggMC4xODUsLTMuNDE1YzAuMTQ5LC0wLjY1OCAwLjM3NCwtNS44IDAuMzkzLC02LjU5OGMwLjEwOSwtNC41MzcgMC4wMDcsLTQuNTMxIDAuMDk2LC05LjA2NWMwLjAxNywtMC44NzkgMC4xODQsLTMuNDk3IC0wLjE3MywtNC44MzJjLTAuMTE2LC0wLjQzMyAtMC4xNjQsLTEuMDg4IC0wLjg0NCwtMi40MDljLTAuMDk1LC0wLjE4NSAtMC45NzMsLTEuMjI2IC0xLjc3MSwtMS42MDZjLTEuOTMyLC0wLjkyMSAtMy4zNTYsLTAuNDU3IC0zLjc0OCwtMC4zNDRjLTEuMDY3LDAuMzA3IC0xLjEwOSwwLjQxMiAtMS4zMDcsMC41MzljLTAuMzYyLDAuMjM0IC0xLjY3MiwxLjM3IC0yLjM0MSwzLjI0NmMtMC40NzYsMS4zMzQgLTAuNzc5LDIuMDg2IC0wLjgyNSwzLjgxNmMtMC4wMDUsMC4xODMgLTAuMDgzLDAuMTc3IC0wLjA5NSwzLjc3OWMtMC4wMTQsNC40ODUgMC40NjgsMTIuODQ1IDAuNzIsMTQuMDk4YzAuMTA5LDAuNTQyIDAuMTU3LDAuNTM1IDAuMjA5LDIuNDcyYzAuMDM0LDEuMjY1IC0wLjI0MSwxLjY3MSAtMC43NSwxLjg0OGMtMC4yNjQsMC4wOTIgLTEuODEyLDAuMTA2IC00LjA1MSwwLjEwOWMtNi4wODcsMC4wMDcgLTYuMTI2LDAuMDE1IC02LjU5OSwtMC4zNTNjLTAuOTIyLC0wLjcxOSAtMC41MzIsLTIuMzQ3IC0wLjQ5LC0yLjUwNmMwLjIwMSwtMC43NTggMC4wNjksLTEuNTEgMC4yMjgsLTIuMjE2YzAuNDU3LC0yLjAyOCAwLjU1NCwtMTUuMDA5IDAuNTI0LC0xNi44OTFjLTAuMDM2LC0yLjIzMyAtMC4wNzMsLTAuNzY4IC0wLjIxMywtMS42OGMtMC4xNDcsLTAuOTUzIC0wLjAxNSwtMC44MSAtMC4yMjksLTEuNTljLTAuMzMzLC0xLjIxNSAtMC45NTUsLTIuNzMyIC0xLjI1OCwtMy4xNTVjLTAuOTk2LC0xLjM5MyAtMS43NzksLTEuODc2IC00LjE1NSwtMS43MzdjLTAuMTU1LDAuMDA5IC0wLjYzLDAuMDM3IC0xLjgyLDAuNjU4Yy0wLjI0NiwwLjEyOSAtMS41MjIsMS4wNzkgLTIuMTQxLDIuMjA5Yy0wLjE1MywwLjI4IC0wLjIsMC4yOSAtMC4zNDgsMC44MjhjLTAuMDE1LDAuMDU0IC0wLjIzOCwwLjYxNiAtMC40OSwxLjUwOGMtMC4yMjEsMC43ODEgLTAuNzE3LDQuNjA4IC0wLjY3NSw4LjU4OWMwLjAwOCwwLjc5MiAwLjM3LDkuMTczIDAuNDExLDkuOTYyYzAuMDksMS43MjQgMC4zODEsMi42ODMgMC40MzcsMy40NzhjMC4xNjIsMi4yNjggLTAuMTE1LDIuMzcgLTAuMzgsMi41MTFjLTAuNzU5LDAuNDA0IC0wLjc5NCwwLjM5NSAtNC4zODIsMC4zOWMtNi4xMzUsLTAuMDA4IC02LjE4OSwwLjAzMSAtNi42MjQsLTAuMzE3Yy0wLjE1NiwtMC4xMjUgLTAuNDU3LC0wLjE3NCAtMC4zODksLTEuNjQ1YzAuMDcyLC0xLjU2NCAwLjIyNCwtMi4yNjMgMC41MjksLTMuNTI1YzAuMjY1LC0xLjA5MiAwLjExMywtNS45NTggMC4xMjQsLTExLjE3YzAuMDIsLTkuMTUzIDAuMDM4LC0xMy45NzIgLTAuMTA5LC0xNC43MjNjLTAuMTc5LC0wLjkxMyAtMC40NjksLTIuOTI2IC0wLjUwNSwtNC4wMzdjLTAuMDM1LC0xLjA4MSAwLjA4NCwtMS4zOTIgMC40MTYsLTEuNTU1YzAuMTkzLC0wLjA5NCAwLjE3NiwtMC4xMTMgMC4zNiwtMC4yMTRjMC40MTcsLTAuMjI4IDMuNzgxLC0xLjA4MiA0LjMxMiwtMS4xMzljMC43OTksLTAuMDg1IDIuMzM0LC0wLjU0NSAzLjEzLC0wLjYxNWMyLjM0NCwtMC4yMDYgMi41MzEsMC4zNzEgMi41MzEsMC4zNzFjMCwwIDAuMjUxLDAuMzE1IDAuMjY3LDAuMzQ2YzAuMTUzLDAuMjgzIDAuMTI2LDEuMzAxIDAuMTA0LDIuMThjLTAuMDI0LDAuOTg1IC0wLjU2MywxLjg5NiAtMC43NTMsMy4wMjNjLTAuMDgzLDAuNDk0IC0wLjE0OSwwLjc0MiAwLjM0MiwwLjY1N2MwLjE4MiwtMC4wMzIgMC4yODgsLTAuMzEzIDAuMzAxLC0wLjM0N2MwLjEyNCwtMC4zMyAwLjEzNSwtMC4zMjEgMC4yOTMsLTAuNjRjMC4wNSwtMC4xMDEgMC41ODUsLTEuNDQ4IDEuMDA2LC0yLjA1OGMwLjQxOCwtMC42MDYgMS4yOCwtMS4zNjYgMS41MTgsLTEuNjA0WiIgc3R5bGU9ImZpbGw6I2ZkN2Q1OTsiLz48cGF0aCBkPSJNMjE2Ljg3MSwxMDUuMzczYzAuMDU5LDAuMDQ4IDMuODUxLDIuMjEgNi44MjEsOC44OTNjMC40MDQsMC45MDkgMC4yMzIsMC45NiAwLjU1NiwxLjg5M2MwLjA2MywwLjE4MiAwLjEwMywwLjU2IDAuMTA4LDAuNjFjMC4wOTksMC45NTMgMC41NTQsMi41NCAwLjU5LDQuMDczYzAuMDA1LDAuMjAxIDAuMDc2LDQuMzMyIC0wLjA3NSw1LjAwMWMtMC4yNzcsMS4yMjggLTAuMDUzLDEuMjY2IC0wLjM5NywyLjQ3M2MtMC4yMiwwLjc3MSAtMC4wOSwwLjc5MyAtMC4zMTUsMS41NjJjLTAuMTgyLDAuNjI0IC0wLjEsMC42MzkgLTAuMzE1LDEuMjQ1Yy0wLjE4NiwwLjUyNSAtMC4xNDksMC41MzYgLTAuNDAyLDEuMDNjLTAuMTUxLDAuMjk0IC0wLjQxNSwxLjA5NiAtMC40NDcsMS4xOTJjLTAuMDI5LDAuMDg5IC0wLjA0OSwwLjE4MiAtMC4wOTUsMC4yNjNjLTAuMDI1LDAuMDQ0IC0wLjA4NywwLjA1OCAtMC4xMTMsMC4xMDFjLTAuMDQ4LDAuMDc5IC0wLjA1MSwwLjE4MiAtMC4xMDQsMC4yNThjLTAuMDM1LDAuMDUgLTAuMTExLDAuMDU5IC0wLjE0OCwwLjEwN2MtMC4xNjMsMC4yMTIgLTIuMzIsNy4xNjkgLTEyLjAyMyw5LjA3NmMtMC42MzQsMC4xMjUgLTAuNjM5LDAuMDQ1IC0xLjI2MiwwLjE5NGMtMC4yMzksMC4wNTcgLTMuNDYyLDAuMjE2IC00LjY1NiwtMC4wODhjLTAuOTIzLC0wLjIzNSAtMC45NjIsLTAuMDE0IC0xLjg2OCwtMC4zMTdjLTAuNzksLTAuMjY0IC0xLjY4MiwtMC41MzEgLTEuODY0LC0wLjYyNWMtMC40OTksLTAuMjU5IC0wLjQ5MiwtMC4yNjkgLTEuMjk1LC0wLjUzNWMtMC4zMzksLTAuMTEzIC0wLjMwNCwtMC4xNzQgLTAuNjI5LC0wLjMwMmMtMC4wODYsLTAuMDM0IC0wLjE3OCwtMC4wNTggLTAuMjU4LC0wLjEwNWMtMC4wNDQsLTAuMDI2IC0wLjA2MiwtMC4wODMgLTAuMTA2LC0wLjEwOWMtMC4wNzksLTAuMDQ3IC0wLjE3OSwtMC4wNTUgLTAuMjU1LC0wLjEwN2MtMC4wNSwtMC4wMzQgLTAuMDYzLC0wLjEwNiAtMC4xMTEsLTAuMTQyYy0wLjI0NywtMC4xODYgLTAuNzYzLC0wLjQ0MiAtMC44MzEsLTAuNDc2Yy0wLjI3NywtMC4xMzggLTAuNjE5LC0wLjQ2OSAtMC42OTUsLTAuNTE4Yy0wLjA3NCwtMC4wNDggLTAuMTY5LC0wLjA2IC0wLjIzOSwtMC4xMTRjLTAuMDUzLC0wLjA0MSAtMC4wNjcsLTAuMTE4IC0wLjExOCwtMC4xNjFjLTAuMDY0LC0wLjA1MyAtMC4xNTgsLTAuMDY0IC0wLjIyLC0wLjExOWMtMC4wNTMsLTAuMDQ3IC0wLjA2NSwtMC4xMjkgLTAuMTE3LC0wLjE3N2MtMC4xNzIsLTAuMTYyIC0wLjI2MSwtMC4xNzEgLTAuMzg5LC0wLjI2MWMtNy4zNSwtNS4yMjYgLTYuMzk3LC0xNy41MzUgLTYuMzM1LC0xOC4zMzZjMC4xOTEsLTIuNDU4IDAuNjgxLC00LjE1NCAwLjg1MiwtNS4wMjJjMC4xMDUsLTAuNTM0IDAuMTU0LC0wLjUxMyAwLjcwNCwtMi4xN2MwLjAzLC0wLjA4OSAwLjA1MywtMC4xODIgMC4wOTgsLTAuMjY1YzAuMDIyLC0wLjA0MSAwLjA3NiwtMC4wNiAwLjA5OCwtMC4xMDFjMC4xMzYsLTAuMjQ5IDAuMDcsLTAuMjcxIDAuMjExLC0wLjUyNWMwLjAyMywtMC4wNDIgMC4wNzMsLTAuMDY1IDAuMDk1LC0wLjEwOGMwLjA4NCwtMC4xNjcgMC4xMzMsLTAuMzQ5IDAuMjE4LC0wLjUxNWMwLjAyMywtMC4wNDUgMC4wOCwtMC4wNjUgMC4xMDMsLTAuMTA5YzAuMDg0LC0wLjE2NiAwLjEyNiwtMC4zNSAwLjIxMiwtMC41MTVjMC4wMjQsLTAuMDQ1IDAuMDg2LC0wLjA2MSAwLjExMiwtMC4xMDVjMC4wNDgsLTAuMDc5IDAuMDU0LC0wLjE4IDAuMTA1LC0wLjI1NmMwLjAzMywtMC4wNDkgMC4xMDUsLTAuMDYxIDAuMTQsLTAuMTA4YzAuMDI0LC0wLjAzMyAwLjEzLC0wLjQ1MiAwLjg3MywtMS40MDdjMi42MTcsLTMuMzY0IDYuNDksLTUuNDQ0IDcuMSwtNS42ODljMC40NjMsLTAuMTg2IDAuNDM2LC0wLjI0OSAwLjkyMSwtMC4zNzNjMC40ODgsLTAuMTI0IDAuNDU2LC0wLjIxNCAwLjk0NSwtMC4zMDZjMC4xOTcsLTAuMDM3IDAuNCwtMC4wNDQgMC41OTQsLTAuMDkzYzAuMTI2LC0wLjAzMiAwLjIzNywtMC4xMDggMC4zNiwtMC4xNDhjMC4wNDcsLTAuMDE2IDAuMjE4LC0wLjA3MiAwLjYxMSwtMC4xMDZjMC42NDQsLTAuMDU2IDAuNjQsLTAuMDU3IDEuMjY1LC0wLjIwNGMwLjcxNiwtMC4xNjkgMy44OTgsLTAuMTk1IDQuNjgxLC0wLjAwNGMwLjc3OCwwLjE5IDAuNzg5LDAuMDk4IDEuNTc4LDAuMjQxYzAuMTAyLDAuMDE4IDAuMDg2LDAuMDc3IDEuMjUzLDAuMzA3YzAuOTA4LDAuMTc5IDIuMDY0LDAuNzkxIDIuMjM0LDAuODgxYzAuMjQ5LDAuMTMyIDAuMjY2LDAuMDgzIDAuNTE0LDAuMjI1YzAuODEsMC40NjEgMS40OTgsMC43NjggMS42MDksMC44NDdjMC4wNTIsMC4wMzggMC4wNywwLjExMSAwLjEyLDAuMTUyWm0tMi4xNzksMTkuNzIyYy0wLjEwMywtMy44MjEgMC4xNDIsLTEzLjk5MSAtNy45MzMsLTEzLjk1M2MtNS41MjQsMC4wMjYgLTcuNjU0LDYuMjcyIC03LjU0MywxNC4wOThjMC4wNDMsMy4wNDQgMS4yNDIsMTAuMDQxIDguMTU4LDkuODk0YzQuOTM1LC0wLjEwNSA3LjQyLC02LjIxNyA3LjMxOCwtMTAuMDM4WiIgc3R5bGU9ImZpbGw6I2ZkN2Q1OTsiLz48cGF0aCBkPSJNMTg0LjAzNCwxMTAuODZjMC4wMDcsMC4zIC0wLjA4MywwLjQwMyAtMC4zNTgsMC44MjZjLTAuMzksMC42MDIgLTEuNzc0LDAuNTE5IC0zLjUzNiwtMC4xNzdjLTUuMzA1LC0yLjA5NCAtMTAuODI2LC0xLjMxNyAtMTAuNzE2LDIuMTQ2YzAuMTI4LDQuMDAzIDMuNjEyLDMuNzIzIDUuNzQ3LDQuNTg3YzIuNTM3LDEuMDI3IDguNTE0LDMuNTI3IDEwLjcxNiw5Ljc2NWMwLjg1MSwyLjQxMiAwLjYxMiw0LjM0NyAwLjU2NSw0LjcyNWMtMC4xMTEsMC44OTQgLTAuNjA1LDYuODQgLTcuNTk3LDkuNTQ0Yy0xLjAzMiwwLjM5OSAtMi44NDQsMC43MjEgLTMuMzY0LDAuODI0Yy0wLjU3NCwwLjExNCAtNC45ODIsMC40ODggLTguMTY0LC0wLjI2MmMtMC45MiwtMC4yMTcgLTEuMzQ4LC0wLjE2NiAtMi41MDEsLTAuNjc2Yy0xLjEzNSwtMC41MDIgLTIuODY5LC0xLjYzNCAtMy45ODgsLTIuMzk3Yy0wLjY4NywtMC40NTcgLTEuMTIzLC0xLjIwOCAtMS4xOCwtMi4wMzFjLTAuMDcyLC0xLjA1IC0wLjE1MiwtMi42MTkgLTAuMTQ4LC00LjJjMC4wMDEsLTAuNTI0IDAuMDk5LC0xLjcwNyAxLjA4MSwtMS43NDNjMC44MDksLTAuMDMgMS4xMDYsLTAuMTY1IDIuNDA1LDAuNzY3YzAuMDk3LDAuMDcgMS4xNjcsMC44MjggMy4zNTcsMS44NDljNS4wMTMsMi4zMzcgOC44NjQsMi4wMyAxMC4xNjYsLTAuMjI5YzAuMjI1LC0wLjM5IDAuMzU0LC0wLjY2MyAwLjM3OSwtMS41OTJjMC4wODgsLTMuMjI0IC0zLjA0NywtNC4zODIgLTQuNTc2LC01LjEwOWMtMS4yNjYsLTAuNjAyIC0xLjIzOSwtMC42NDYgLTIuNTE2LC0xLjIyN2MtMC4yMjQsLTAuMTAyIC00LjAxNSwtMS45NSAtNS4yMzcsLTIuOTUxYy0xLjA0LC0wLjg1MSAtMS40NTcsLTEuMTg3IC0yLjYxNiwtMi43MWMtMS41MTYsLTEuOTkyIC0yLjMxNiwtNS4zODIgLTEuODc5LC03Ljg4N2MwLjA3NiwtMC40MzYgMC4yNjIsLTMuODE1IDMuNzI5LC02Ljk5M2MxLjE1NiwtMS4wNiAxLjI1OCwtMC45MTIgMi42NDQsLTEuNjY5YzAuODksLTAuNDg3IDQuNjUxLC0xLjU5NyA3Ljg0MSwtMS4yNTdjMy4wMzcsMC4zMjMgNS40NzksMS4wMTggNy4wODYsMS41OTFjMS4zMjYsMC40ODMgMi4yNTEsMS42OTIgMi4zNywzLjA5OGMwLjEzMywxLjMxMSAwLjI3NCwyLjgyNSAwLjI4OCwzLjM4N1oiIHN0eWxlPSJmaWxsOiNmZDdkNTk7Ii8+PHBhdGggZD0iTTE0NC43MDksMTQwLjg2NmMtMC4wMDIsMC4xMDEgMC4wMTIsMC4yMDIgMC4wMSwwLjMwM2MtMC4wMzMsMS40NDUgLTIuMTk1LDEuMTkgLTIuNjk5LDEuMTkxYy0wLjg3NywwLjAwMiAtNy42MjksMC4wMDUgLTcuODE0LC0wLjAwOWMtMS4yNCwtMC4wOTYgLTEuMjEsLTEuMzM4IC0xLjIwNiwtMS40OTFjMC4wNDIsLTEuNzI5IDAuMjg4LC0xLjcwNSAwLjM2NiwtMy40MzNjMC4yMjIsLTQuODc3IDAuNTE4LC04LjcyMiAwLjU0NiwtMTAuNjQxYzAuMTU0LC0xMC44ODkgMC4wNzIsLTEwLjg4OSAtMC40NDYsLTE2LjU2M2MtMC4xMDEsLTEuMTAyIDAuMDMyLC0xLjIzNiAtMC4zMjIsLTIuODAzYy0wLjQ1NSwtMi4wMTggMC40NjYsLTEuOTg2IDMuODc1LC0yLjgyNWMwLjkyMywtMC4yMjcgMC45MzYsLTAuMTUxIDEuNTU5LC0wLjMyOGMwLjM1NywtMC4xMDIgMy43NDgsLTEuMDY4IDQuNjM2LC0wLjI1MWMxLjAzMywwLjk1MSAwLjY1OCwxLjg2NiAwLjU5Nyw1LjI1N2MtMC4wMzMsMS44MjcgLTAuMjA2LDIuNTkyIC0wLjI1NiwyLjgxMmMtMC4wMTEsMC4wNDkgLTAuMTM0LDAuNTkxIC0wLjEzLDAuNjEzYzAuMDA3LDAuMDQ1IDAuMDgzLDAuNTI5IDAuNDcxLDAuMzI3YzAuMTU4LC0wLjA4MiAwLjcxNCwtMi41NTUgMi43MjgsLTUuMzczYzAuNzcsLTEuMDc4IDAuNzQ3LC0xLjI2NyAyLjQ3NSwtMi44N2MwLjE0MywtMC4xMzMgMC44NDQsLTAuNzgzIDIuMDU4LC0xLjMxNGMxLjUxNSwtMC42NjMgMi41MjIsLTAuNjgxIDIuNzQ5LC0wLjY4NmMxLjI5LC0wLjAyMyA0LjE4NywwLjQwMyA0LjE2OCwyLjQyNmMtMC4wMDMsMC4zNDYgLTAuMTQxLDEuMTQ1IC0wLjE3OCwxLjU1NGMtMC4xMzUsMS41MzEgLTAuNTY5LDMuMzk1IC0wLjYxMiwzLjc1Yy0wLjI2NSwyLjE4MyAtMC40MDIsMi44OTUgLTIuMDgxLDIuNDI4Yy0xLjI2OSwtMC4zNTIgLTYuMTA3LDAuNjEyIC04LjcxMiw1LjQ1M2MtMS43MzMsMy4yMjIgLTIuMjksNi44OSAtMi40NjEsOC4yODVjLTAuNDM1LDMuNTUxIC0wLjE2NCw5LjIzNiAwLjY3OSwxNC4xODRaIiBzdHlsZT0iZmlsbDojZmQ3ZDU5OyIvPjxwYXRoIGQ9Ik0xMy4zODQsMTMxLjg5N2MwLjMyNCwwLjMyMSAwLjMwNCwwLjMzOCAwLjY2OSwwLjYxNGMwLjYwNiwwLjQ1OCAxLjIyMywwLjg3OSAyLjIxLDEuMjU3YzEuMzk1LDAuNTM0IDEuOTE0LDAuNDY4IDIuMjI1LDAuNTQzYzAuODI5LDAuMiAwLjgzMywwLjE4NiAyLjgxOSwwLjE3MmMwLjc5NCwtMC4wMDYgMi42MjIsLTAuMjk3IDIuNzc0LC0wLjM0M2MwLjAwMywtMC4wMDEgMS4wMDUsLTAuMzI0IDEuNTg1LC0wLjU0OWMwLjgsLTAuMzExIDIuNDYyLC0xLjA5MSAyLjg5NCwtMS4zOTRjMC41MjcsLTAuMzcgMC41MjcsLTAuMzY2IDAuNTczLC0wLjM5N2MwLjk5MiwtMC42NjcgMS44LC0wLjk1NCAyLjE3MywtMC45MDZjMC4yNTMsMC4wMzIgMC42MjEsMC4xODQgMC43NjIsMC40NDFjMC4yMjYsMC40MTEgMC42NjcsMS4wNDMgMC41OTIsMi4wMjZjLTAuMDIzLDAuMjk5IC0wLjA0NiwwLjU3MiAtMC4yMDcsMC44NzZjLTAuMDIyLDAuMDQxIC0xLjM2NiwyLjU4MyAtMi4xODMsMy4zODhjLTAuMjExLDAuMjA3IC0xLjg4NSwxLjg1NSAtMi4wNSwyLjAxNmMtMC43NDUsMC43MyAtMS42MDUsMS4zMzUgLTEuNjk0LDEuNDAzYy0wLjAzMywwLjAyNSAtMC45MDUsMC40NzMgLTEuMjA3LDAuNjVjLTAuMDk2LDAuMDU2IC0wLjEwNCwwLjAzNyAtMS4yNDgsMC42MTljLTAuMTY3LDAuMDg1IC0xLjE2MiwwLjQwMyAtMS44MDUsMC41MzdjLTAuMzgyLDAuMDggLTIuMDMsMC41NCAtMi44MzMsMC41NjFjLTMuNjI5LDAuMDk1IC01LjE1LC0wLjIyNiAtNS45NDksLTAuMjgyYy0wLjkzOCwtMC4wNjYgLTMuODgyLC0xLjA2NyAtNC4wMDMsLTEuMTM0Yy0wLjA0MywtMC4wMjQgLTMuMjExLC0yLjAxNiAtMy4zOTMsLTIuMTg1Yy0wLjAyOCwtMC4wMjYgLTEuNDMsLTEuMjg5IC0yLjAxMiwtMi4wNDhjLTAuNjU5LC0wLjg1OSAtMS44NjgsLTIuOTM3IC0xLjk0MSwtMy4xMDljLTEuNjI2LC0zLjgxNyAtMS44NDgsLTUuMTY5IC0yLjA3MSwtNy44NzNjLTAuMjI3LC0yLjc2NSAwLjE4NywtNi4yMTggMC43MDcsLTguNzE3YzAuNDU3LC0yLjE5NiAyLjM3NCwtNi4xOTcgMi40MTMsLTYuMjhjMC4xMiwtMC4yNTQgMy45NDUsLTkuNDQzIDE1LjMwNSwtOS4zNjdjMS41OTgsMC4wMTEgMTIuMzg3LDAuMjE1IDE0LjA1OCwxMy4xMThjMC4yNjYsMi4wNTMgMC4zNTUsNC44NSAtMC4wODcsNi4yMWMtMC4wNTEsMC4xNTcgLTAuNDg4LDEuNSAtMC44NzcsMS44NjZjLTAuMDYxLDAuMDU3IC0wLjE1MSwwLjA3NSAtMC4yMTgsMC4xMjZjLTAuMjU1LDAuMTkzIC0wLjI0OCwwLjI4NSAtMS4yOTksMC4yODljLTYuMjU5LDAuMDI0IC0xMC45MTYsMC4wMiAtMTQuMDksMC4wMmMtMy40MywtMCAtNS4xMjcsMC4wMDUgLTUuMjM5LDAuMDU2Yy0wLjA1MiwwLjAyNCAtMC4wNzQsMC4wODcgLTAuMTE1LDAuMTI3Yy0wLjExMSwwLjEwOSAtMC4yNCwwLjA5NyAtMC4xNjIsMS4yOThjMC4wNTEsMC43OTYgMC45NTksMy4zNjMgMS4wNDIsMy41MDVjMC4xNDYsMC4yNDkgMC44ODUsMS42OTIgMS4yNjYsMi4xOTZjMC4yNzYsMC4zNjQgMC4yOTIsMC4zNDMgMC42MTYsMC42NjlabTQuNTIxLC0xMy42OTdjMC44MzksLTAuMDA1IDEuNzMzLC0wLjAxIDMuNDA1LC0wLjAxNmMxLjU4OSwtMC4wMDUgMi4yMTcsLTAuMjIzIDIuMDYyLC0yLjk2MmMtMC4xNzIsLTMuMDE5IC0xLjkzNCwtNS41NjMgLTQuNTgzLC02LjA1NGMtMC4zMSwtMC4wNTcgLTMuNTYzLC0wLjYwMSAtNi4yMTIsMy4wNDJjLTAuNDc0LDAuNjUxIC0xLjkxMywyLjMyOCAtMi4xMTYsNS4xOWMtMC4wMDQsMC4wNTEgMC4wMiwwLjM2MSAwLjI1NCwwLjU4NWMwLjI4OSwwLjI3NiAwLjk3NywwLjIyNyAzLjQwMSwwLjIyNWMxLjkyOCwtMC4wMDEgMi44MjIsLTAuMDA1IDMuNzg5LC0wLjAxMVoiIHN0eWxlPSJmaWxsOiNmZDdkNTk7Ii8+PHBhdGggZD0iTTEwOC41MzUsMTMxLjg5N2MwLjMyNCwwLjMyMSAwLjMwNCwwLjMzOCAwLjY2OSwwLjYxNGMwLjYwNiwwLjQ1OCAxLjIyMywwLjg3OSAyLjIxLDEuMjU3YzEuMzk1LDAuNTM0IDEuOTE0LDAuNDY4IDIuMjI1LDAuNTQzYzAuODI5LDAuMiAwLjgzMywwLjE4NiAyLjgxOSwwLjE3MmMwLjc5NCwtMC4wMDYgMi42MjIsLTAuMjk3IDIuNzc0LC0wLjM0M2MwLjAwMywtMC4wMDEgMS4wMDUsLTAuMzI0IDEuNTg1LC0wLjU0OWMwLjgsLTAuMzExIDIuNDYyLC0xLjA5MSAyLjg5NCwtMS4zOTRjMC41MjcsLTAuMzcgMC41MjcsLTAuMzY2IDAuNTczLC0wLjM5N2MwLjk5MiwtMC42NjcgMS44LC0wLjk1NCAyLjE3MywtMC45MDZjMC4yNTMsMC4wMzIgMC42MjEsMC4xODQgMC43NjIsMC40NDFjMC4yMjYsMC40MTEgMC42NjcsMS4wNDMgMC41OTIsMi4wMjZjLTAuMDIzLDAuMjk5IC0wLjA0NiwwLjU3MiAtMC4yMDcsMC44NzZjLTAuMDIyLDAuMDQxIC0xLjM2NiwyLjU4MyAtMi4xODMsMy4zODhjLTAuMjExLDAuMjA3IC0xLjg4NSwxLjg1NSAtMi4wNSwyLjAxNmMtMC43NDUsMC43MyAtMS42MDUsMS4zMzUgLTEuNjk0LDEuNDAzYy0wLjAzMywwLjAyNSAtMC45MDUsMC40NzMgLTEuMjA3LDAuNjVjLTAuMDk2LDAuMDU2IC0wLjEwNCwwLjAzNyAtMS4yNDgsMC42MTljLTAuMTY3LDAuMDg1IC0xLjE2MiwwLjQwMyAtMS44MDUsMC41MzdjLTAuMzgyLDAuMDggLTIuMDMsMC41NCAtMi44MzMsMC41NjFjLTMuNjI5LDAuMDk1IC01LjE1LC0wLjIyNiAtNS45NDksLTAuMjgyYy0wLjkzOCwtMC4wNjYgLTMuODgyLC0xLjA2NyAtNC4wMDMsLTEuMTM0Yy0wLjA0MywtMC4wMjQgLTMuMjExLC0yLjAxNiAtMy4zOTMsLTIuMTg1Yy0wLjAyOCwtMC4wMjYgLTEuNDMsLTEuMjg5IC0yLjAxMiwtMi4wNDhjLTAuNjU5LC0wLjg1OSAtMS44NjgsLTIuOTM3IC0xLjk0MSwtMy4xMDljLTEuNjI2LC0zLjgxNyAtMS44NDgsLTUuMTY5IC0yLjA3MSwtNy44NzNjLTAuMjI3LC0yLjc2NSAwLjE4NywtNi4yMTggMC43MDcsLTguNzE3YzAuNDU3LC0yLjE5NiAyLjM3NCwtNi4xOTcgMi40MTMsLTYuMjhjMC4xMiwtMC4yNTQgMy45NDUsLTkuNDQzIDE1LjMwNSwtOS4zNjdjMS41OTgsMC4wMTEgMTIuMzg3LDAuMjE1IDE0LjA1OCwxMy4xMThjMC4yNjYsMi4wNTMgMC4zNTUsNC44NSAtMC4wODcsNi4yMWMtMC4wNTEsMC4xNTcgLTAuNDg4LDEuNSAtMC44NzcsMS44NjZjLTAuMDYxLDAuMDU3IC0wLjE1MSwwLjA3NSAtMC4yMTgsMC4xMjZjLTAuMjU1LDAuMTkzIC0wLjI0OCwwLjI4NSAtMS4yOTksMC4yODljLTYuMjU5LDAuMDI0IC0xMC45MTYsMC4wMiAtMTQuMDksMC4wMmMtMy40MywtMCAtNS4xMjcsMC4wMDUgLTUuMjM5LDAuMDU2Yy0wLjA1MiwwLjAyNCAtMC4wNzQsMC4wODcgLTAuMTE1LDAuMTI3Yy0wLjExMSwwLjEwOSAtMC4yNCwwLjA5NyAtMC4xNjIsMS4yOThjMC4wNTEsMC43OTYgMC45NTksMy4zNjMgMS4wNDIsMy41MDVjMC4xNDYsMC4yNDkgMC44ODUsMS42OTIgMS4yNjYsMi4xOTZjMC4yNzYsMC4zNjQgMC4yOTIsMC4zNDMgMC42MTYsMC42NjlabTQuNTIxLC0xMy42OTdjMC44MzksLTAuMDA1IDEuNzMzLC0wLjAxIDMuNDA1LC0wLjAxNmMxLjU4OSwtMC4wMDUgMi4yMTcsLTAuMjIzIDIuMDYyLC0yLjk2MmMtMC4xNzIsLTMuMDE5IC0xLjkzNCwtNS41NjMgLTQuNTgzLC02LjA1NGMtMC4zMSwtMC4wNTcgLTMuNTYzLC0wLjYwMSAtNi4yMTIsMy4wNDJjLTAuNDc0LDAuNjUxIC0xLjkxMywyLjMyOCAtMi4xMTYsNS4xOWMtMC4wMDQsMC4wNTEgMC4wMiwwLjM2MSAwLjI1NCwwLjU4NWMwLjI4OSwwLjI3NiAwLjk3NywwLjIyNyAzLjQwMSwwLjIyNWMxLjkyOCwtMC4wMDEgMi44MjIsLTAuMDA1IDMuNzg5LC0wLjAxMVoiIHN0eWxlPSJmaWxsOiNmZDdkNTk7Ii8+PHBhdGggZD0iTTE2MS40NzQsNDkuMDU1Yy0wLjU0OSwtMC43MzMgLTEsLTEuNDU4IC0xLjQ2OSwtMS43M2MtMC4yNywtMC4xNTcgLTIuMDcyLC0yLjUwNSAtMi4yMzUsLTMuMzk1Yy0xLjEzNSwtNi4yMTUgNy4zNjIsLTEuNzk2IDEzLjM1OSwwLjU2NWM0LjQ1NywxLjc1NSAzLjY2NCwzLjYwMiAwLjk3Niw3LjM5NmMtNC43ODgsNi43NTkgLTYuNDI2LDkuMDI2IC04LjQ4OSw3LjQ1OWMtMS42MiwtMS4yMyAtMC41NTIsLTQuMTkxIC0wLjU1OCwtNC44NjdjLTAuMDE5LC0yLjIyMyAtMC41MzgsLTQuMDM0IC0xLjU4NCwtNS40MjlaIiBzdHlsZT0iZmlsbDojOTI0ZmFjOyIvPjxwYXRoIGQ9Ik0xNjEuOTAyLDU0LjIyNmMtMC4xNDQsMC40MTggLTAuNTMzLDAuODkxIC0wLjg5MSwxLjEwOWMtNS44MjcsMy41NTYgLTYuMDM1LDUuMjI0IC0xMi43NTEsMTIuMTA2Yy0xMi4zMDIsMTIuNjA2IC0yMy40MzgsMTMuNjUgLTI0LjU3LDEzLjkwOWMtMy4wNzQsMC43MDMgLTguOTA4LDAuMiAtMTAuMzU2LC0wLjIyYy03LjU2NSwtMi4xOTEgLTExLjM1NCwtNS4yNSAtMTQuNTQ2LC0xMS43NWMtMC42ODUsLTEuMzk1IC0xLjU0OCwtNS41ODkgLTEuNTUsLTUuNjA4Yy0wLjgyLC01Ljk3NSAtMC4xMDgsLTguNDY2IC0wLjA4NSwtOC41OTljMC4yOTksLTEuNjkxIC0wLjk4NSwtMC43NjIgLTMuMTk5LDAuNDU4Yy00Ljg1NCwyLjY3MyAtNC40MjYsLTIuNzk4IC0zLjk4LC0zLjc4NWMxLjkzNiwtNC4yNzkgNS4zMzMsLTUuNDQ5IDguMjA2LC03LjA5NGMxLjA3NiwtMC42MTYgMC41OSwtMi43OTMgMy4xMjksLTkuNjMzYzMuMDYyLC04LjI0OCA1LjMxLC0xMS4xMDMgNy4zNiwtMTQuMjcyYzMuMjY5LC01LjA1NCA3Ljk4OSwtMTAuMDk1IDExLjcwMywtMTMuMTAxYzExLjY3OCwtOS40NTEgMjMuMjQ1LC04LjU2NiAyNy41MTgsLTYuMDk1YzguMTg1LDQuNzM0IDUuMDY2LDEzLjk2OCA0LjUzMiwxNS43NjVjLTAuNjMsMS42IC0wLjgyMiwyLjc1NSAtMy4xNzIsNi4zNDZjLTAuNzc5LDEuMTkgLTQuNzQyLDcuMjQ1IC0xMi40MzMsMTIuNzAyYy05LjM0NSw2LjYzMSAtMjAuMzgsMTAuNjI5IC0yMy42NTcsMTEuODk4Yy0xLjQ2LDAuNTY1IC0xLjE2OSwwLjkwNSAtMS4zMDQsMS43MDVjLTAuOTMxLDUuNTQ0IC0wLjY3NSw4LjE5MiAwLjcwNCwxMS45NzhjMi4xODQsNS45OTggMTIuNjI4LDEwLjUyNyAyNS4zNTQsMS43MjJjNi45MTIsLTQuNzgyIDExLjc2NiwtMTEuNTcxIDE5LjU1NCwtMTUuMjM0YzAuMjAxLC0wLjA5NSAwLjg0NSwtMC40MDMgMS40MywtMC4yNzVjMC40OTUsMC4xMDkgMC43ODgsMC4zMSAwLjkyLDAuNDhjMC40MzUsMC41NTkgMC43NTMsMC44ODEgMS4wODEsMS40OThjMC41NDgsMS4wMzMgMC44NjEsMS43NTkgMS4wMzYsMi40MWMwLjE3OSwwLjY2NiAwLjExOSwxLjEzMSAtMC4wMzYsMS41OFptLTQ2LjUzNSwtMTguMjI1Yy0wLjk5NSwyLjczNCAtMS4wNDEsMi44OTUgLTAuOCwyLjkyOWMwLjI2MSwwLjAzNyAxMC4xNjUsLTMuNzg1IDE1LjQwNSwtNy4zODNjMC4yNSwtMC4xNzEgOC4wNDUsLTQuNzE5IDExLjcxMSwtMTIuNTVjMy4yNDMsLTYuOTI4IDAuNjczLC0xMC4yMzYgLTEuNDc5LC0xMC41MjhjLTEzLjE4MiwtMS43ODcgLTIzLjY4OSwyMy4zMjcgLTI0LjgzNywyNy41MzFaIiBzdHlsZT0iZmlsbDojZmQ3ZDU5OyIvPjwvc3ZnPg==" alt="Emerson">
    <div class="topbar-sep"></div>
    <div class="topbar-app">podcast editor</div>
  </div>
  <div class="topbar-spacer"></div>
  <button class="btn-topbar" onclick="document.getElementById('jingleModal').classList.add('open')">⚡ Vinheta</button>
  <div class="topbar-status">
    <div class="status-dot" id="statusDot"></div>
    <span id="statusText">ocioso</span>
  </div>
  <div class="topbar-user" onclick="doLogout()">
    <div class="user-avatar" id="navAvatar">—</div>
    <div class="user-name"   id="navName">—</div>
  </div>
</div>

<div id="app">

  <div id="hero">
    <div class="hero-overline">Editor profissional</div>
    <h1 class="hero-headline">Corte.<br>Edite.<br><span>Publique.</span></h1>
    <p class="hero-sub">
      Transcrição com Whisper, análise editorial com IA e exportação
      profissional — sem limite de tamanho de arquivo.
    </p>

    <div id="upload-zone">
      <input type="file" id="fileInput" accept="video/*,audio/*">
      <div class="wave-viz">
        <div class="w"></div><div class="w"></div><div class="w"></div>
        <div class="w"></div><div class="w"></div><div class="w"></div>
        <div class="w"></div><div class="w"></div><div class="w"></div>
        <div class="w"></div><div class="w"></div><div class="w"></div>
        <div class="w"></div>
      </div>
      <div class="upload-cta">Arraste o arquivo aqui</div>
      <div class="upload-hint">MP4 · MOV · MKV · M4A · MP3 &nbsp;·&nbsp; <strong>sem limite de tamanho</strong></div>
    </div>
  </div>

  <div id="processing-panel">
    <div class="proc-wrap">
      <div class="proc-num" id="procNum">01</div>
      <div class="proc-meta">
        <div class="proc-stage" id="procStage">Iniciando</div>
        <div class="proc-title" id="procTitle">Preparando...</div>
        <div class="proc-msg"   id="procMsg">Aguarde</div>
      </div>
    </div>
    <div class="prog-track">
      <div class="prog-fill" id="progFill" style="width:0%"></div>
    </div>
    <div class="prog-meta-row">
      <span id="progLabel">—</span>
      <span id="progPct">0%</span>
    </div>
    <div class="steps-list">
      <div class="step-item" id="step-upload">
        <div class="step-ico" id="si-upload">01</div>
        <div class="step-lbl">Upload e validação do arquivo</div>
      </div>
      <div class="step-item" id="step-whisper">
        <div class="step-ico" id="si-whisper">02</div>
        <div class="step-lbl">Transcrição com Whisper</div>
      </div>
      <div class="step-item" id="step-claude">
        <div class="step-ico" id="si-claude">03</div>
        <div class="step-lbl">Análise editorial com IA</div>
      </div>
    </div>
  </div>

  <div id="analysis-section">

    <div class="sec-div"><div class="sec-n">01</div><div class="sec-lbl">Episódio</div><div class="sec-line"></div></div>
    <div class="ep-grid">
      <div class="ep-main">
        <div class="ep-eyebrow">Título sugerido</div>
        <div class="ep-title" id="epTitle">—</div>
        <div class="ep-alts" id="epAlts"></div>
        <div class="ep-summary" id="epSummary"></div>
      </div>
      <div class="stats-panel">
        <div class="stats-head">Métricas</div>
        <div class="stat-row"><div class="stat-val" id="stBruto">—</div><div class="stat-lbl">Duração bruta</div></div>
        <div class="stat-row"><div class="stat-val" id="stEdit">—</div><div class="stat-lbl">Duração editada</div></div>
        <div class="stat-row"><div class="stat-val" id="stPPM">—</div><div class="stat-lbl">Palavras / min</div></div>
        <div class="stat-row"><div class="stat-val" id="stPausa">—</div><div class="stat-lbl">Pausas longas</div></div>
      </div>
    </div>

    <div class="sec-div"><div class="sec-n">02</div><div class="sec-lbl">Destaques</div><div class="sec-line"></div></div>
    <div class="hl-grid">
      <div class="hl-card">
        <div class="hl-eye">Melhor clip para redes</div>
        <div class="hl-time" id="hlClipTime">—</div>
        <div class="hl-desc" id="hlClipDesc">—</div>
      </div>
      <div class="hl-card">
        <div class="hl-eye">Frase destaque</div>
        <div class="hl-quote" id="hlQuote">—</div>
        <div class="hl-desc"  id="hlQuoteTs">—</div>
      </div>
    </div>

    <div class="sec-div"><div class="sec-n">03</div><div class="sec-lbl">Capítulos</div><div class="sec-line"></div></div>
    <div class="chaps-grid" id="chapsGrid"></div>

    <div id="probSec" style="display:none">
      <div class="sec-div"><div class="sec-n">04</div><div class="sec-lbl">Problemas detectados</div><div class="sec-line"></div></div>
      <div id="probList"></div>
    </div>

    <div class="sec-div"><div class="sec-n" id="cutsSN">05</div><div class="sec-lbl">Cortes sugeridos</div><div class="sec-line"></div></div>
    <div class="cuts-bar">
      <div class="cuts-info"><strong id="cutsTotal">0</strong> segmentos · <strong id="cutsSel">0</strong> selecionados</div>
      <div class="cuts-btns">
        <button class="btn-sm" onclick="selectAll(true)">Selecionar todos</button>
        <button class="btn-sm" onclick="selectAll(false)">Limpar</button>
      </div>
    </div>
    <div id="cutsTable"></div>

    <div class="sec-div" id="txDiv" style="display:none"><div class="sec-n">06</div><div class="sec-lbl">Transcrição</div><div class="sec-line"></div></div>
    <div class="tx-wrap" id="txWrap" style="display:none">
      <div class="tx-inner" id="txBox"></div>
    </div>

    <div class="sec-div"><div class="sec-n">—</div><div class="sec-lbl">Exportar</div><div class="sec-line"></div></div>
    <div class="export-panel">
      <div class="exp-opts">
        <label class="opt-lbl"><input type="checkbox" id="makeClip" checked> Gerar clip 60–90s</label>
        <label class="opt-lbl"><input type="checkbox" id="makeTx"   checked> Exportar transcrição</label>
      </div>
      <button class="btn-export" id="exportBtn" onclick="exportJob()">Executar cortes e exportar →</button>
    </div>

  </div>

  <div id="downloads-section">
    <div class="sec-div"><div class="sec-n">✓</div><div class="sec-lbl">Pronto para download</div><div class="sec-line"></div></div>
    <div class="dl-grid" id="dlGrid"></div>
  </div>

</div>
</div><!-- /main-app -->

<!-- JINGLE MODAL -->
<div class="modal-ov" id="jingleModal">
  <div class="modal-box">
    <div class="modal-hd">
      <div>
        <div class="modal-title">Configurar Vinheta</div>
        <div class="modal-sub">Adicionada automaticamente em todos os episódios</div>
      </div>
      <button class="modal-close" onclick="document.getElementById('jingleModal').classList.remove('open')">×</button>
    </div>
    <div class="modal-bd">
      <div class="f-field">
        <label class="f-label">Abertura</label>
        <div class="f-row" onclick="document.getElementById('jOpen').click()">
          <input type="file" id="jOpen" accept="video/*,audio/*" onchange="setFn(this,'fno')">
          <span class="f-name" id="fno">Nenhum arquivo</span>
          <span class="f-browse">Selecionar</span>
        </div>
      </div>
      <div class="f-field">
        <label class="f-label">Fechamento</label>
        <div class="f-row" onclick="document.getElementById('jClose').click()">
          <input type="file" id="jClose" accept="video/*,audio/*" onchange="setFn(this,'fnc')">
          <span class="f-name" id="fnc">Nenhum arquivo</span>
          <span class="f-browse">Selecionar</span>
        </div>
      </div>
      <button class="btn-save" onclick="saveJingles()">Salvar vinhetas</button>
    </div>
  </div>
</div>

<script>
/* ══ AUTH ════════════════════════════════════════════════════ */
const USERS = {
  'admin':   { pass: 'emerson2026', role: 'admin',  name: 'Admin' },
  'editor1': { pass: 'editor123',   role: 'editor', name: 'Editor' },
};

let currentUser = null;
let pendingUser = null;

function doLogin() {
  const u = document.getElementById('loginUser').value.trim();
  const p = document.getElementById('loginPass').value;
  const errEl = document.getElementById('loginError');
  errEl.textContent = '';

  if (!u || !p) { errEl.textContent = 'Preencha usuário e senha.'; return; }

  const user = USERS[u];
  if (!user || user.pass !== p) {
    errEl.textContent = 'Usuário ou senha inválidos.';
    return;
  }

  currentUser = { username: u, ...user };

  if (user.role === 'admin') {
    enterApp();
  } else {
    // Editor: precisa de aprovação do admin
    showApprovalWait();
  }
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    if (document.getElementById('login-screen').style.display !== 'none') doLogin();
  }
});

function showApprovalWait() {
  document.getElementById('login-screen').style.display = 'none';
  const sc = document.getElementById('approval-screen');
  sc.style.display = 'flex';

  // Preenche badge do usuário aguardando
  const initials = currentUser.name.split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
  document.getElementById('approvalAvatar').textContent = initials;
  document.getElementById('approvalName').textContent   = currentUser.name;
  document.getElementById('approvalRole').textContent   = 'Editor · aguardando aprovação';

  // Simula: admin vê os botões de aprovação/negação
  // Em produção real, isso seria uma rota separada para o admin
  // Aqui: se clicar na tela de aprovação com Ctrl+A, aparece painel admin
  pendingUser = currentUser;

  document.addEventListener('keydown', adminShortcut);
}

function adminShortcut(e) {
  if (e.ctrlKey && e.key === 'a') {
    e.preventDefault();
    // Mostra controles de admin (simula admin ver a tela)
    document.getElementById('approvalActions').style.display = 'grid';
    document.getElementById('approvalPending').style.display = 'none';
  }
}

function approveAccess() {
  document.removeEventListener('keydown', adminShortcut);
  document.getElementById('approval-screen').style.display = 'none';
  enterApp();
}

function denyAccess() {
  document.removeEventListener('keydown', adminShortcut);
  currentUser = null; pendingUser = null;
  document.getElementById('approval-screen').style.display = 'none';
  document.getElementById('login-screen').style.display = 'flex';
  document.getElementById('loginError').textContent = 'Acesso negado pelo administrador.';
  document.getElementById('loginUser').value = '';
  document.getElementById('loginPass').value = '';
}

function enterApp() {
  document.getElementById('login-screen').style.display    = 'none';
  document.getElementById('approval-screen').style.display = 'none';
  document.getElementById('main-app').style.display        = 'block';

  const initials = currentUser.name.split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
  document.getElementById('navAvatar').textContent = initials;
  document.getElementById('navName').textContent   = currentUser.name;
}

function doLogout() {
  if (!confirm('Sair do sistema?')) return;
  currentUser = null;
  document.getElementById('main-app').style.display     = 'none';
  document.getElementById('login-screen').style.display = 'flex';
  document.getElementById('loginUser').value = '';
  document.getElementById('loginPass').value = '';
  document.getElementById('loginError').textContent = '';
  // Reset app state
  location.reload();
}

/* ══ APP ═════════════════════════════════════════════════════ */
let jobId = null, poll = null, cuts = [];

function fmt(s) {
  s = s||0;
  return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`;
}

function setStatus(t, live=false) {
  document.getElementById('statusText').textContent = t;
  document.getElementById('statusDot').className = 'status-dot'+(live?' live':'');
}

const fileInput  = document.getElementById('fileInput');
const uploadZone = document.getElementById('upload-zone');

fileInput.addEventListener('change', e => { if(e.target.files[0]) upload(e.target.files[0]) });
uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag') });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag');
  const f = e.dataTransfer.files[0]; if(f) upload(f);
});

const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB por chunk

async function upload(file) {
  document.getElementById('hero').style.display = 'none';
  document.getElementById('processing-panel').style.display = 'block';
  setStatus('carregando…', true);
  setStep('upload','active');
  setProcState('01','Upload',`Enviando ${file.name}…`,`${(file.size/1024/1024).toFixed(1)} MB`);
  setProgress(2,'Iniciando envio');

  // 1. Criar job_id no servidor
  const initRes = await fetch('/upload-init', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({filename: file.name})
  });
  const {job_id} = await initRes.json();
  jobId = job_id;

  // 2. Dividir arquivo em chunks de 5MB e enviar
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end   = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const fd = new FormData();
    fd.append('job_id',       job_id);
    fd.append('chunk_index',  i);
    fd.append('total_chunks', totalChunks);
    fd.append('filename',     file.name);
    fd.append('chunk',        chunk, file.name);

    await fetch('/upload-chunk', {method:'POST', body: fd});

    // Progresso: 2% → 8% durante upload
    const pct = Math.round(2 + ((i + 1) / totalChunks) * 6);
    setProgress(pct, `Enviando… ${i+1}/${totalChunks} partes`);
  }

  setStep('upload','done');
  setProgress(8, 'Upload concluído — processando…');
  poll = setInterval(doPoll, 2000);
}

function setStep(id, state) {
  const item = document.getElementById(`step-${id}`);
  const ico  = document.getElementById(`si-${id}`);
  if(!item||!ico) return;
  item.className = 'step-item'+(state==='active'?' active':state==='done'?' done':'');
  ico.className  = 'step-ico'+(state==='active'?' spinning':state==='done'?' ok':'');
  if(state==='done') ico.textContent = '✓';
}

function setProcState(num, stage, title, msg) {
  document.getElementById('procNum').textContent   = num;
  document.getElementById('procNum').className     = 'proc-num lit';
  document.getElementById('procStage').textContent = stage;
  document.getElementById('procTitle').textContent = title;
  document.getElementById('procMsg').textContent   = msg;
}

function setProgress(pct, lbl) {
  document.getElementById('progFill').style.width  = pct+'%';
  document.getElementById('progPct').textContent   = pct+'%';
  document.getElementById('progLabel').textContent = lbl||'—';
}

async function doPoll() {
  if(!jobId) return;
  const res = await fetch(`/status/${jobId}`);
  const job = await res.json();
  setProgress(job.progress||0, job.msg);

  if(job.progress>=10 && job.progress<60) {
    setStep('whisper','active');
    setProcState('02','Transcrição','Whisper processando o áudio…','Pode levar alguns minutos');
    setStatus('transcrevendo',true);
  } else if(job.progress>=60) {
    setStep('whisper','done');
    setStep('claude','active');
    setProcState('03','Análise IA','Claude analisando o conteúdo…','Identificando cortes e destaques');
    setStatus('analisando',true);
  }

  if(job.status==='ready') {
    clearInterval(poll);
    setStep('claude','done');
    setStatus('pronto');
    showAnalysis(job);
  } else if(job.status==='done') {
    clearInterval(poll);
    setStatus('exportado');
    showDownloads(job);
  } else if(job.status==='error') {
    clearInterval(poll);
    setStatus('erro');
    setProcState('!!','Erro',job.msg||'Erro inesperado','Tente novamente');
  }
}

function showAnalysis(job) {
  document.getElementById('processing-panel').style.display = 'none';
  document.getElementById('analysis-section').style.display = 'block';
  const a=job.analysis||{}, ep=a.episodio||{}, st=a.estatisticas||{};

  document.getElementById('epTitle').textContent   = ep.titulo_sugerido||'Sem título';
  document.getElementById('epSummary').textContent = ep.resumo||'';
  document.getElementById('epAlts').innerHTML = (ep.titulos_alternativos||[]).map(t=>
    `<div class="ep-alt" onclick="document.getElementById('epTitle').textContent=this.textContent">${t}</div>`
  ).join('');

  document.getElementById('stBruto').textContent  = Math.round((job.duration||0)/60)+'min';
  document.getElementById('stEdit').textContent   = Math.round((st.recomendacao_duracao_final||0)/60)+'min';
  document.getElementById('stPPM').textContent    = st.palavras_por_minuto||'—';
  document.getElementById('stPausa').textContent  = st.pausas_longas||'0';

  const mc=a.melhor_clip||{};
  document.getElementById('hlClipTime').textContent = mc.inicio!=null?`${fmt(mc.inicio)} → ${fmt(mc.fim)}`:'—';
  document.getElementById('hlClipDesc').textContent = mc.motivo||'';

  const fq=a.frase_destaque||{};
  document.getElementById('hlQuote').textContent  = fq.texto||'—';
  document.getElementById('hlQuoteTs').textContent= fq.inicio!=null?`${fmt(fq.inicio)} → ${fmt(fq.fim)}`:'';

  document.getElementById('chapsGrid').innerHTML = (a.capitulos||[]).map(c=>`
    <div class="chap-card">
      <div class="chap-ts">${fmt(c.inicio)}</div>
      <div class="chap-title">${c.titulo}</div>
    </div>`).join('');

  const probs=a.problemas_detectados||[];
  if(probs.length) {
    document.getElementById('probSec').style.display='block';
    document.getElementById('cutsSN').textContent='05';
    const lbl={silencio_longo:'Silêncio longo',vicio_linguagem:'Vício de linguagem',audio_ruim:'Áudio ruim',repeticao:'Repetição'};
    document.getElementById('probList').innerHTML = probs.map(p=>`
      <div class="prob-row">
        <div class="prob-type">${lbl[p.tipo]||p.tipo}</div>
        <div class="prob-desc">${p.descricao}</div>
        <div class="prob-at">${fmt(p.inicio)}</div>
      </div>`).join('');
  } else {
    document.getElementById('cutsSN').textContent='04';
  }

  cuts=(a.cortes||[]).map(c=>({...c,sel:c.tipo!=='cortar'}));
  renderCuts();

  if(job.transcript) {
    document.getElementById('txDiv').style.display='flex';
    document.getElementById('txWrap').style.display='block';
    document.getElementById('txBox').textContent=job.transcript;
  }
}

function renderCuts() {
  const sel=cuts.filter(c=>c.sel).length;
  document.getElementById('cutsTotal').textContent=cuts.length;
  document.getElementById('cutsSel').textContent=sel;

  document.getElementById('cutsTable').innerHTML=cuts.map((c,i)=>{
    const dur=Math.round(c.duracao||(c.fim-c.inicio));
    const prio=c.prioridade||2;
    return `<div class="cut-row${c.sel?'':' removed'}" id="cr-${i}">
      <div class="cut-bar ${c.sel?c.tipo:'removed'}"></div>
      <div class="cut-tc">
        <div class="tc-r">${fmt(c.inicio)} → ${fmt(c.fim)}</div>
        <div class="tc-d">${dur}s</div>
      </div>
      <div class="cut-body">
        <div class="cut-tags">
          <span class="ctag ${c.tipo}">${c.tipo}</span>
          <span class="ctag ${c.energia||'media'}">${c.energia||'media'}</span>
        </div>
        <div class="cut-reason">${c.justificativa||''}</div>
      </div>
      <div class="cut-prio"><div class="pip p${prio}"></div></div>
      <div class="cut-save"><div class="save-v">−${dur}s</div><div class="save-l">economia</div></div>
      <div class="cut-tog-cell">
        <button class="cut-toggle${c.sel?' on':''}" onclick="toggleCut(${i})">${c.sel?'✓':'−'}</button>
      </div>
    </div>`;
  }).join('');
}

function toggleCut(i)    { cuts[i].sel=!cuts[i].sel; renderCuts() }
function selectAll(val)  { cuts=cuts.map(c=>({...c,sel:val})); renderCuts() }

async function exportJob() {
  const approved=cuts.filter(c=>c.sel);
  if(!approved.length){alert('Selecione pelo menos um segmento.');return}
  const btn=document.getElementById('exportBtn');
  btn.disabled=true; btn.textContent='Exportando…';
  document.getElementById('analysis-section').style.display='none';
  document.getElementById('processing-panel').style.display='block';
  setProcState('04','Export','FFmpeg executando os cortes…','Montando episódio final');
  setProgress(72,'Processando segmentos');
  setStatus('exportando',true);
  await fetch(`/approve/${jobId}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cuts:approved,make_clip:document.getElementById('makeClip').checked})});
  poll=setInterval(async()=>{
    const r=await fetch(`/status/${jobId}`); const j=await r.json();
    setProgress(j.progress||72,j.msg||'…');
    if(j.status==='done'){clearInterval(poll);showDownloads(j)}
    else if(j.status==='error'){clearInterval(poll);setStatus('erro');setProcState('!!','Erro',j.msg,'Tente novamente')}
  },2000);
}

function showDownloads(job) {
  document.getElementById('processing-panel').style.display='none';
  document.getElementById('downloads-section').style.display='block';
  setStatus('pronto');
  const items=[
    {type:'video',      n:'01',icon:'🎬',title:'Podcast Editado',  sub:'Vídeo final com vinheta',       avail:!!job.output_video},
    {type:'clip',       n:'02',icon:'📱',title:'Clip Horizontal',  sub:'16:9 — melhor momento',         avail:!!job.output_clip},
    {type:'reel',       n:'03',icon:'🎞',title:'Reel Vertical',    sub:'9:16 — Instagram / TikTok',     avail:!!job.output_reel},
    {type:'transcript', n:'04',icon:'📝',title:'Transcrição',      sub:'Texto completo do episódio',    avail:!!job.output_transcript},
    {type:'srt',        n:'05',icon:'💬',title:'Legenda',          sub:'Arquivo .srt para YouTube',     avail:!!job.output_srt},
  ];
  document.getElementById('dlGrid').innerHTML=items.filter(i=>i.avail).map(i=>`
    <div class="dl-card">
      <div class="dl-n">${i.n}</div>
      <div class="dl-ico">${i.icon}</div>
      <div class="dl-title">${i.title}</div>
      <div class="dl-sub">${i.sub}</div>
      <a class="btn-dl" href="/download/${jobId}/${i.type}" download>⬇ Baixar</a>
    </div>`).join('');
}

function setFn(input,id){document.getElementById(id).textContent=input.files[0]?input.files[0].name:'Nenhum arquivo'}
async function saveJingles(){
  const o=document.getElementById('jOpen').files[0];
  const c=document.getElementById('jClose').files[0];
  const ups=[];
  if(o){const fd=new FormData();fd.append('type','open');fd.append('file',o);ups.push(fetch('/upload-jingle',{method:'POST',body:fd}))}
  if(c){const fd=new FormData();fd.append('type','close');fd.append('file',c);ups.push(fetch('/upload-jingle',{method:'POST',body:fd}))}
  await Promise.all(ups);
  document.getElementById('jingleModal').classList.remove('open');
}

document.getElementById('jingleModal').addEventListener('click',e=>{
  if(e.target===document.getElementById('jingleModal')) document.getElementById('jingleModal').classList.remove('open');
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
