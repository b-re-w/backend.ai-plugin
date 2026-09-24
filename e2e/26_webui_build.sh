# Build the user's WebUI fork (backend.ai-webui, develop) in WSL and serve it from the E2E webserver.
# Sources are copied from the Windows checkout; node_modules stay on the WSL side between builds.
set -euo pipefail   # a failed step must stop before the webserver is switched
D=$(dirname "$0"); . $D/env.sh; R=$W/run
NODE_DIR=$HOME/.cache/labgpu-node
WB=$HOME/labgpu-webui
export PATH=$NODE_DIR/bin:$PATH COREPACK_ENABLE_DOWNLOAD_PROMPT=0

if [ ! -x $NODE_DIR/bin/node ]; then
  tarball=$(curl -s https://nodejs.org/dist/latest-v24.x/ | grep -o 'node-v24[0-9.]*-linux-x64.tar.xz' | head -1)
  mkdir -p $NODE_DIR
  curl -sSL https://nodejs.org/dist/latest-v24.x/$tarball | tar -xJ -C $NODE_DIR --strip-components=1
fi
echo "node $(node -v)"
corepack enable --install-directory $NODE_DIR/bin >/dev/null 2>&1 || true

mkdir -p $WB
rsync -a --delete --exclude node_modules --exclude .git --exclude build --exclude dist \
  --exclude '*/node_modules' $SRC/backend.ai-webui/ $WB/
# A Windows checkout has CRLF endings; scripts with a shebang (scripts/copy-config.js) then fail.
grep -rlIZ "$(printf '\r')" --exclude-dir=node_modules $WB | xargs -0 -r sed -i 's/\x0d$//'
cd $WB
[ -f config.toml ] || cp configs/default.toml config.toml 2>/dev/null || true
pnpm --version
pnpm install --frozen-lockfile --reporter=silent 2>&1 | tail -5
pnpm run build 2>&1 | tail -15
test -f build/web/index.html && ls build/web/assets | grep -q '^index-.*\.js$'   # the React bundle exists
# The webserver serves this build; keep the config the bundled WebUI shipped with.
cp $W/bai/src/ai/backend/web/static/config.toml build/web/config.toml
sed -i "s@^static_path = .*@static_path = \"$WB/build/web\"@" $R/webserver.conf
grep -n "^static_path" $R/webserver.conf
ps -eo pid,args | grep -E "backend.ai: web" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null; sleep 2
cd $R && nohup setsid $PY -m ai.backend.cli web start-server -f webserver.conf > webserver.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/ | grep -q 200 && break; sleep 2; done
echo "webserver: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/)"
