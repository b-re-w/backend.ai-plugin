# Storage proxy (vfolders) for the WebUI's folder pages: vfs volume "volume1" under the agent mount path.
D=$(dirname "$0"); . $D/env.sh; R=$W/run; cd $R
mkdir -p $R/vfroot/local/volume1
cp $W/bai/configs/storage-proxy/halfstack.toml storage-proxy.toml
sed -i 's/\r$//' storage-proxy.toml
sed -i 's/port = 2379 }/port = 8120 }/' storage-proxy.toml
# Must equal the secret the manager holds for this proxy (etcd volumes/proxies/local/secret).
sed -i 's/secret = "some-secret-shared-with-manager"/secret = "some-secret-shared-with-storage-proxy"/' storage-proxy.toml
sed -i "s@ssl-cert = \"configs/@ssl-cert = \"$W/bai/configs/@; s@ssl-privkey = \"configs/@ssl-privkey = \"$W/bai/configs/@" storage-proxy.toml
sed -i "s@^path = \"vfolder/local/volume1\"@path = \"$R/vfroot/local/volume1\"@" storage-proxy.toml
grep -nE "port = 8120|secret =|ssl-cert|^path" storage-proxy.toml
ps -eo pid,args | grep -E "backend.ai: storage" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null; sleep 2
nohup setsid $PY -m ai.backend.cli storage start-server -f storage-proxy.toml > storage-proxy.log 2>&1 < /dev/null &
for i in $(seq 1 40); do curl -s -o /dev/null http://127.0.0.1:6021/ && break; sleep 2; done
echo "storage-proxy client api: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:6021/)"
sed 's/\x1b\[[0-9;]*m//g' storage-proxy.log | grep -E "ERROR|Traceback|Error:" | head -5 | cut -c1-200
