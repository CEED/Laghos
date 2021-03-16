NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -offline -writesol -romsvds -romsrhs
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -romgs -rdimx 4 -rdimv 20 -rdime 16 -soldiff -romsrhs
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -romgs -rdimx 4 -rdimv 20 -rdime 16 -romhrprep -nsamx 18 -nsamv 3401 -nsame 128 -romsrhs
    $LAGHOS_SERIAL -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -romgs -rdimx 4 -rdimv 20 -rdime 16 -romhr -nsamx 18 -nsamv 3401 -nsame 128 -romsrhs
    ;;
  4)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -romgs -rdimx 4 -rdimv 20 -rdime 16 -romhrprep -sfacv 40 -sface 40 -qdeim -romsrhs
    $LAGHOS_SERIAL -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -online -romgs -rdimx 4 -rdimv 20 -rdime 16 -romhr -sfacv 40 -sface 40 -qdeim -romsrhs
    ;;
  5)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.12 -s 7 -pa -restore -rdimx 4 -rdimv 20 -rdime 16 -soldiff -romsrhs
    ;;
esac
