NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -offline -writesol -romsvds -no-romsns -romos
    ;;
  2)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -romgs -rdimx 2 -rdimv 6 -rdime 2 -soldiff -no-romsns -romos
    ;;
  3)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -romgs -rdimx 2 -rdimv 6 -rdime 2 -rdimfv 9 -rdimfe 11 -romhrprep -nsamx 96 -nsamv 320 -nsame 64 -no-romsns -romos
    $LAGHOS_SERIAL -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -romgs -rdimx 2 -rdimv 6 -rdime 2 -rdimfv 9 -rdimfe 11 -romhr -nsamx 96 -nsamv 320 -nsame 64 -no-romsns -romos
    ;;
  4)
    $LAGHOS -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -restore -rdimx 2 -rdimv 6 -rdime 2 -soldiff -no-romsns -romos
    ;;
esac
