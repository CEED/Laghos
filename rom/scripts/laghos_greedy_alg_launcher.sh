# Check if greedy algorithm has ended
while [ "$?" -eq 0 ]
do
    "$@"
done
exit 0
