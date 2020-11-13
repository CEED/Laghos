NUM_PARALLEL_PROCESSORS=8
testNames=(offline online)
absoluteFOMOptions="-p 3 -m data/box01_hex.mesh -rs 1 -tf 0.2 -cfl 0.05 -pa"
absoluteFOMTol="1e-7"
absoluteFOMTolParallel="1e-7"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="2"
case $subTestNum in
  1)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.2 -cfl 0.05 -pa -offline -writesol -romsvds -rostype load
    ;;
  2)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.2 -cfl 0.05 -pa -online -rdimx 3 -rdimv 7 -rdime 5 -romhr -nsamx 6 -nsamv 448 -nsame 10 -soldiff -rostype load
    ;;
esac
