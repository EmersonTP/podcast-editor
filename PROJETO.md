# Podcast Editor — Estado do Projeto

## URLs
- **Produção:** (configurar após deploy no Railway)
- **Repositório:** (configurar após push no GitHub)

## Variáveis de ambiente (Railway)
| Variável | Valor |
|---|---|
| `ANTHROPIC_API_KEY` | sua chave da Anthropic |
| `WHISPER_MODEL` | `base` (opções: tiny/base/small/medium/large) |
| `UPLOAD_DIR` | `/data/uploads` |
| `PROC_DIR` | `/data/processed` |
| `DATA_DIR` | `/data` |

## Volumes Railway
- Mount path: `/data` (obrigatório — salva uploads, processados e jobs.json)

## Pipeline
1. Upload vídeo → Whisper transcreve → Claude analisa → usuário aprova cortes → FFmpeg exporta
2. Entregáveis: vídeo editado + clip 60–90s para redes + transcrição completa

## Decisões tomadas
- Stack: FastAPI + Railway (mesmo padrão do brandbook)
- Transcrição: Whisper local (sem custo, sem API externa)
- Análise editorial: Claude claude-sonnet-4-6
- Vinheta: upload manual pela interface, fica salva no servidor
- Sem autenticação por enquanto (uso interno)

## Próximos passos
- [ ] Deploy Railway
- [ ] Testar com vídeo real de 30 minutos
- [ ] Ajustar WHISPER_MODEL conforme qualidade/velocidade desejada
- [ ] Adicionar suporte a Google Drive (upload direto via link)
