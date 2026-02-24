# Download missing/placeholder listening TRN & VAL HDFs for subjects 01–06, 08
subjects=(01 02 03 04 05 06 08)
baseurl="https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri/raw/master/responses"
for s in "${subjects[@]}"; do
  for split in trn val; do
    url="$baseurl/subject${s}_reading_fmri_data_${split}.hdf"
    out="data/responses/subject${s}_reading_fmri_data_${split}.hdf"
    if [[ ! -f "$out" ]] || file "$out" | grep -q 'ASCII text'; then
      echo "Downloading $out"
      wget -q --show-progress -O "$out" "$url" || echo "Failed to download $out (skipping)"
    else
      echo "Already HDF5, skipping: $out"
    fi
  done
done

# Verify file types and sizes
for s in "${subjects[@]}"; do
  for split in trn val; do
    out="data/responses/subject${s}_reading_fmri_data_${split}.hdf"
    if [[ -f "$out" ]]; then
      file "$out"
      ls -lh "$out"
    else
      echo "Missing $out"
    fi
  done
done