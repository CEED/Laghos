NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -pa"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1e-14"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="1.5"
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -pa -offline -writesol -romsvds -sdim 10000
    ;;
  2)
    $LAGHOS -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -pa -online -rdimx 26 -rdimv 107 -rdime 90 -romhr -nsamx 3328 -nsamv 4802 -nsame 2304 -soldiff
    ;;
esac
