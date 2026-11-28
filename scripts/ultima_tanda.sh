#!/usr/bin/env bash
# The last batch on the DGX (2026-11-28): changes G, A and I, development and exam, in one go.
#
#     nohup bash scripts/ultima_tanda.sh > ~/ultima_tanda.log 2>&1 &
#     tail -f ~/ultima_tanda.log            # to watch it (Ctrl+C does not stop it)
#
# Two tracks at once (2 + 2 requests; the course's server allows 5):
#
#   G then I: the triage v2, v3 and v3p on the lorazepam development batches; the version for
#             the exam is chosen by the rule below; the exam of v2 and the chosen one, ONCE;
#             then change I (the triage's help to the MedDRA coder) with the final coder of A
#             and the chosen triage.
#   A:        the final 27B coder (rlm/coder27b.py) on 50 training problems (development: we do
#             not iterate any more, it is a check), then the phase-1 exam (test 200 + OOD 100),
#             ONCE, without and with reasoning.
#
# The rule to choose the triage, fixed BEFORE seeing any result (written 2026-11-28 10:30):
#   v3p if its detection over both development batches is higher than v3's AND it sends to a
#   person at most twice as many mentions of the normal batch as v2; otherwise v3.
#   Detection first (a lost case nobody sees again), but a queue that doubles is not usable.
#
# Every step resumes where it stopped if it is launched again. Nothing here changes the code.
set -u
cd "$(dirname "$0")/.."

log() { echo "[$(date +%H:%M:%S)] $*"; }

key=$(grep '^ARCA_REMOTE_LLM_API_KEY=' .env | cut -d= -f2-)
url=$(grep '^ARCA_REMOTE_LLM_URL=' .env | cut -d= -f2-)
url=${url:-http://0.0.0.0:8001/v1}
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $key" "$url/models")
if [ "$code" != "200" ]; then
  log "el 27B no responde en $url (código $code): no se lanza nada"
  exit 1
fi
log "27B listo en $url"
for f in lote_lorazepam_dev lote_lorazepam_test extra_lorazepam_dev extra_lorazepam_test; do
  [ -f "data/lorazepam/$f.jsonl" ] || { log "falta data/lorazepam/$f.jsonl"; exit 1; }
done

track_g_i() {
  set -e
  log "G · desarrollo: v2, v3 y v3p en los dos lotes"
  for v in v2 v3 v3p; do
    for l in lorazepam extra_lorazepam; do
      uv run python -m triage.run --version "$v" --por mencion --lote "$l"
    done
  done
  uv run python -m triage.lotes --versiones v2 v3 v3p
  chosen=$(uv run python - <<'EOF'
import json
d = json.load(open("reports/triage_lorazepam.json", encoding="utf-8"))["dev"]["versiones"]
det = {v: d[v]["deteccion_ambos_lotes"]["k"] for v in d}
queue = {v: d[v]["a_revision_lote_normal"]["menciones"] for v in d}
ok = det["v3p"] > det["v3"] and queue["v3p"] <= 2 * queue["v2"]
print("v3p" if ok else "v3")
EOF
)
  log "G · versión elegida por la regla: $chosen"
  echo "$chosen" > reports/triaje_elegido.txt
  log "G · EXAMEN (una vez): v2 y $chosen"
  for v in v2 "$chosen"; do
    for l in lorazepam extra_lorazepam; do
      uv run python -m triage.run --version "$v" --por mencion --lote "$l" --examen
    done
  done
  uv run python -m triage.lotes --versiones v2 "$chosen" --examen
  log "I · la ayuda del triaje ($chosen) al codificador final, sin razonamiento"
  uv run python -m rlm.evaluate_pista --codificador 27b-final --triaje "$chosen"
  log "G e I terminados"
}

track_a() {
  set -e
  log "A · desarrollo: 50 problemas de train con el codificador final"
  uv run python -m rlm.evaluate_remote --conjuntos train --n 50 --codificador final
  log "A · EXAMEN (una vez): test y fuera de distribución, sin y con razonamiento"
  uv run python -m rlm.evaluate_remote --conjuntos test ood --codificador final
  log "A terminado"
}

track_g_i > ~/tanda_g_i.log 2>&1 &
pid_g=$!
track_a > ~/tanda_a.log 2>&1 &
pid_a=$!
log "lanzadas: G+I (~/tanda_g_i.log) y A (~/tanda_a.log)"
wait $pid_g && log "G+I bien" || log "G+I se paró: mira ~/tanda_g_i.log y vuelve a lanzar el script"
wait $pid_a && log "A bien" || log "A se paró: mira ~/tanda_a.log y vuelve a lanzar el script"
log "fin"
