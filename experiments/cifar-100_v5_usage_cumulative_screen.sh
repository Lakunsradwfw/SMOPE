# Run only the cumulative-usage v5 mechanism screen.
#
# This keeps the strict-current vs non-strict comparison but avoids launching
# all usage modes at once.

USAGE_MODES=cumulative bash experiments/cifar-100_v5_usage_mechanism_screen.sh
