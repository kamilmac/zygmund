#!/usr/bin/env bash
# Deploy web/ to gh-pages with cache-busting version queries (?v=<sha>) on every
# cross-file reference, so a browser can never mix resources from two deploys.
set -euo pipefail
cd "$(dirname "$0")"
VER=$(git rev-parse --short HEAD)
SITE=$(mktemp -d)

cp index.html main.js worklet.js worklet-polyfill.js manifest.webmanifest icon-192.png icon-512.png "$SITE/"
cp -R pkg "$SITE/pkg"
rm -f "$SITE/pkg/.gitignore" # wasm-pack regenerates it with '*' — it would exclude pkg from the deploy commit
touch "$SITE/.nojekyll"

sed -i '' "s|\./main\.js|./main.js?v=$VER|" "$SITE/index.html"
sed -i '' "s|'\./worklet\.js'|'./worklet.js?v=$VER'|" "$SITE/main.js"
sed -i '' "s|'\./pkg/zygfred_web_bg\.wasm'|'./pkg/zygfred_web_bg.wasm?v=$VER'|" "$SITE/main.js"
sed -i '' "s|'\./pkg/zygfred_web\.js'|'./pkg/zygfred_web.js?v=$VER'|" "$SITE/main.js"
sed -i '' "s|'\./worklet-polyfill\.js'|'./worklet-polyfill.js?v=$VER'|" "$SITE/worklet.js"
sed -i '' "s|'\./pkg/zygfred_web\.js'|'./pkg/zygfred_web.js?v=$VER'|" "$SITE/worklet.js"

git -C "$SITE" init -q -b gh-pages
git -C "$SITE" add .
git -C "$SITE" commit -qm "deploy $VER"
git -C "$SITE" push -qf git@github.com:kamilmac/zygmund.git gh-pages
rm -rf "$SITE"
echo "deployed $VER"
