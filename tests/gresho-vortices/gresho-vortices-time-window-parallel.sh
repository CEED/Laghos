normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    srun -n 8 -p pdebug laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -offline -ef 0.9999 -normtype "$normtype" -writesol -romsvds -nwin 4 -tw "$BASE_DIR"/tests/gresho-vortices/gresho-vortices-time-window.csv -sdim 800
    ;;
  online)
    srun -n 8 -p pdebug laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -online -normtype "$normtype" -soldiff -romsvds -nwin 4 -twp "$BASE_DIR"/twpTemp.csv
    ;;
  romhr)
    srun -n 8 -p pdebug laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -online -normtype "$normtype" -soldiff -romhr -romsvds -sfacx 50 -sfacv 50 -sface 50 -nwin 4 -twp "$BASE_DIR"/twpTemp.csv
    ;;
  restore)
    srun -n 8 -p pdebug laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -restore -normtype "$normtype" -nwin 4 -twp "$BASE_DIR"/twpTemp.csv
    ;;
esac
