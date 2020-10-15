NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
runAbsoluteFOM="true"
absoluteFOMOptions="-p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -print"
absoluteFOMTol="1e-11"
absoluteFOMTolParallel="1"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="2"
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -offline -writesol -romsvds
    ;;
  2)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -soldiff
    ;;
  3)
    $LAGHOS  -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -soldiff -romhr -nsamx 96 -nsamv 320 -nsame 64
    ;;
  4)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -restore -rdimx 2 -rdimv 6 -rdime 2
    ;;
esac
