# Replace Git LFS pointer files in a `git archive` export with the real objects from backend.ai/.git/lfs.
# Prints the oids that are missing locally (fetch them on Windows: git lfs fetch origin <tag>).
. "$(dirname "$0")/env.sh"
OBJ=$SRC/backend.ai/.git/lfs/objects
filled=0; missing=0
while IFS= read -r f; do
  oid=$(sed -n 's/^oid sha256:\([0-9a-f]\{64\}\)$/\1/p' "$f")
  [ -n "$oid" ] || continue
  src=$OBJ/${oid:0:2}/${oid:2:2}/$oid
  if [ -f "$src" ]; then cp "$src" "$f"; filled=$((filled+1)); else echo "missing $oid ${f#$W/bai/}"; missing=$((missing+1)); fi
done < <(grep -rlI --include='*' '^version https://git-lfs.github.com/spec/v1' $W/bai/src)
echo "LFS: filled $filled, missing $missing"
