NUM_PARALLEL_PROCESSORS=1
testNames=(offline romhr)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.02 -offline -writesol -romsvds -no-romsns -no-romgs -no-romoffset -romst gnat_lspg
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.02 -online -romhrprep -rdimx 3 -rdimv 17 -rdime 7 -rdimfv 25 -rdimfe 12 -ntsamv 1 -ntsame 1 -sfacv 2 -sface 2 -no-romgs -no-romoffset -romst gnat_lspg
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.02 -online -romhr -rdimx 3 -rdimv 17 -rdime 7 -rdimfv 25 -rdimfe 12 -ntsamv 1 -ntsame 1 -sfacv 2 -sface 2 -no-romgs -no-romoffset -romst gnat_lspg -soldiff
    ;;
esac
