NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
runAbsoluteFOM="true"
absoluteFOMOptions="-p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -print"
absoluteFOMTol="1e-10"
absoluteFOMTolParallel="1e-1"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="2"
case $subTestNum in
  1)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -offline -writesol -romsvds
    ;;
  2)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -rdimx 1 -rdimv 4 -rdime 3 -soldiff
    ;;
  3)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -rdimx 1 -rdimv 4 -rdime 3 -romhr -nsamx 6 -nsamv 448 -nsame 10 -soldiff
    ;;
  4)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -restore -rdimx 1 -rdimv 4 -rdime 3
    ;;
esac
