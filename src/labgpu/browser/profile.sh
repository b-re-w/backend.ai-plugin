# Mounted read-only as /etc/profile.d/labgpu-browser-<key>.sh by the labgpu plugins (SPEC 1.14).
# Puts the labgpu xdg-open first on PATH for programs that ignore $BROWSER. One copy per GPU
# plugin key, so a session with several keys runs it more than once; that is harmless.
for _labgpu_dir in /opt/labgpu/browser-*; do
  if [ -x "$_labgpu_dir/xdg-open" ]; then
    case ":$PATH:" in
      *":$_labgpu_dir:"*) ;;
      *) PATH="$_labgpu_dir:$PATH"; export PATH ;;
    esac
    break
  fi
done
unset _labgpu_dir
