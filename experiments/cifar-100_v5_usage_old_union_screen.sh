# Run only the old-union-usage v5 mechanism screen.
#
# This keeps the strict-current vs non-strict comparison but avoids launching
# all usage modes at once.

USAGE_MODES=old_union bash experiments/cifar-100_v5_usage_mechanism_screen.sh
