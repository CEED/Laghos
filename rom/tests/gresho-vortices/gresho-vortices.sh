NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -offline -writesol -romsvds
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -soldiff
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -romhrprep -nsamx 18 -nsamv 3401 -nsame 128 -soldiff
    $LAGHOS_SERIAL -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -romhr -nsamx 18 -nsamv 3401 -nsame 128 -soldiff
    ;;
  4)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -romhrprep -sfacv 40 -sface 40 -soldiff -qdeim
    $LAGHOS_SERIAL -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -rdimx 4 -rdimv 20 -rdime 16 -romhr -sfacv 40 -sface 40 -soldiff -qdeim
    ;;
  5)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -restore -rdimx 4 -rdimv 20 -rdime 16
    ;;
esac
