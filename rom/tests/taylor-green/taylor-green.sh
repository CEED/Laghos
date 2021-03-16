NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -offline -writesol -romsvds -romsrhs
    ;;
  2)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -soldiff -romsrhs
    ;;
  3)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -romhrprep -nsamx 96 -nsamv 320 -nsame 64 -romsrhs
    $LAGHOS_SERIAL -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -romhr -nsamx 96 -nsamv 320 -nsame 64 -romsrhs
    ;;
  4)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -restore -rdimx 2 -rdimv 6 -rdime 2 -soldiff -romsrhs
    ;;
esac
