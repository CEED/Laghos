NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-p 0 -rs 2 -iv -cfl 0.1 -tf 0.02 -pa"
absoluteFOMTol="1e-7"
absoluteFOMTolParallel="1e-7"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="0.8"
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.1 -tf 0.02 -pa -offline -writesol -romsvds -rostype load
    ;;
  2)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.1 -tf 0.02 -pa -online -rdimx 3 -rdimv 20 -rdime 2 -soldiff -rostype load
    ;;
esac
