normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -ef 0.9999 -normtype "$normtype" -nwin 4 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window.csv -writesol -romsvds
    ;;
  online)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -online -nwin 4 -normtype "$normtype" -twp "$BASE_DIR"/twpTemp.csv -soldiff
    ;;
  romhr)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romhr -nwin 4 -normtype "$normtype" -twp "$BASE_DIR"/twpTemp.csv -soldiff
    ;;
  restore)
    srun -n 8 -p pdebug laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -restore -nwin 4 -normtype "$normtype" -twp "$BASE_DIR"/twpTemp.csv
    ;;
esac
