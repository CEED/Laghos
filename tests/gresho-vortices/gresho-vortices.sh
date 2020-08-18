NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $HEADER laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -offline -writesol -romsvds
    ;;
  2)
    $HEADER laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -soldiff
    ;;
  3)
    $HEADER laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -romhr -nsamx 18 -nsamv 3401 -nsame 128 -soldiff
    ;;
  4)
    $HEADER laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -restore -rdimx 4 -rdimv 20 -rdime 16
    ;;
esac
