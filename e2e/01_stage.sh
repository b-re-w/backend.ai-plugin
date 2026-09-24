set -eu
. "$(dirname "$0")/env.sh"
mkdir -p $W/bai
# A release straight from git: LF line endings and real symlinks, no Windows fix-ups needed.
rm -rf $W/bai/src $W/bai/configs $W/bai/fixtures $W/bai/docs
git -C $SRC/backend.ai -c safe.directory='*' -c core.autocrlf=false archive --format=tar "$BAI_REF"   src configs fixtures docs/manager/graphql-reference docker-compose.halfstack-main.yml VERSION requirements.txt   | tar -x -C $W/bai
bash "$(dirname "$0")/fill_lfs.sh"   # git archive exports LFS pointers, not the binaries
rsync -a --delete --exclude '__pycache__' $SRC/backend.ai-plugin/ $W/plugin/
echo "staged Backend.AI $(cat $W/bai/VERSION) into $W"
du -sh $W/bai $W/plugin
