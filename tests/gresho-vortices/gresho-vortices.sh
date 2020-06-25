normtype=${normtype:-"l2"}
case $testtype in
  offline)
    ./laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -offline -normtype "$normtype" -writesol -romsvds
    ;;
  online)
    ./laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -normtype "$normtype" -soldiff
    ;;
  romhr)
    ./laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -romhr -nsamx 18 -nsamv 3401 -nsame 128 -normtype "$normtype" -soldiff
    ;;
  restore)
    ./laghos -p 4 -m "$BASE_DIR"/data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -restore -rdimx 4 -rdimv 20 -rdime 16 -normtype "$normtype"
    ;;
esac
