#!/bin/bash

OUTPUT_FILE="CHANGELOG.md"

# Header changelog
cat <<EOF > "$OUTPUT_FILE"
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - $(date +%F)
EOF

# Inisialisasi kategori
declare -A sections
sections["feat"]="### Added"
sections["fix"]="### Fixed"
sections["refactor"]="### Changed"
sections["chore"]="### Changed"
sections["misc"]="### Misc"

# Buat file sementara kosong
for key in "${!sections[@]}"; do
  > "/tmp/changelog-$key"
done

# Ambil git log
git log --date=short --pretty="%ad - %s" | while IFS= read -r line; do
  # Ambil tanggal dan pesan commit
  tanggal="${line:0:10}"
  pesan="${line:13}"  # Skip "YYYY-MM-DD - "

  # Ambil prefix lowercase tanpa spasi
  prefix=$(echo "$pesan" | sed -E 's/^([[:alpha:]]+)[ ]*:.*/\1/' | tr '[:upper:]' '[:lower:]')

  case "$prefix" in
    feat)
      echo "- ${tanggal:5} - $pesan" >> /tmp/changelog-feat
      ;;
    fix)
      echo "- ${tanggal:5} - $pesan" >> /tmp/changelog-fix
      ;;
    refactor)
      echo "- ${tanggal:5} - $pesan" >> /tmp/changelog-refactor
      ;;
    chore)
      echo "- ${tanggal:5} - $pesan" >> /tmp/changelog-chore
      ;;
    *)
      echo "- ${tanggal:5} - $pesan" >> /tmp/changelog-misc
      ;;
  esac
done

# Tambahkan ke CHANGELOG.md
echo "" >> "$OUTPUT_FILE"
for key in feat fix refactor chore misc; do
  if [[ -s /tmp/changelog-$key ]]; then
    echo "${sections[$key]}" >> "$OUTPUT_FILE"
    cat "/tmp/changelog-$key" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
  fi
done

# Hapus sementara
rm /tmp/changelog-*

echo "✅ CHANGELOG.md berhasil dibuat dari git log!"
