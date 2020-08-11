normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -normtype "$normtype" -writesol -romsvds
    ;;
  online)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -nsamx 12 -nsamv 184 -nsame 30 -normtype "$normtype" -soldiff
    ;;
  romhr)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -romhr -nsamx 4 -nsamv 24 -nsame 32 -normtype "$normtype" -soldiff
    ;;
  restore)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 12 -rdime 16 -normtype "$normtype"
    ;;
esac
