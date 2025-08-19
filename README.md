# Kinova Arm Test

This folder contains a simple test script to verify Kinova arm connection and basic movement functionality.

## Files

- `test_kinova_movement.py` - Main test script for Kinova arm movement

## Usage

### Prerequisites

1. **Python 3.8 Environment**: Make sure you're in the `byovla_kinova_py38` conda environment
2. **Kinova Robot**: Ensure the robot is powered on and accessible at 192.168.2.9:10000

### Running the Test

```bash
# Activate the Python 3.8 environment
conda activate byovla_kinova_py38

# Run the test script
python test_kinova_movement.py
```

## What the Test Does

The script will:

1. **Connect** to the Kinova robot at 192.168.2.9:10000
2. **Get current pose** of the end-effector
3. **Test movements**:
   - Small forward movement (5cm)
   - Return to original position
   - Small upward movement (5cm)
   - Return to original position
4. **Disconnect** from the robot

## Safety Features

- **Small movements**: Only 5cm movements to minimize risk
- **Return to origin**: Always returns to starting position
- **Timeout protection**: 20-second timeout for each movement
- **Error handling**: Comprehensive error checking and reporting

## Expected Output

If successful, you should see:
- Connection established
- Current pose displayed
- 4 movement tests completed
- Robot returning to original position

## Troubleshooting

### Connection Refused
- Check if robot is powered on
- Verify robot IP address (192.168.2.9)
- Check if robot's TCP service is running on port 10000

### Movement Failures
- Ensure workspace is clear
- Check robot error state
- Verify robot is not in emergency stop

## Next Steps

Once this basic test works, you can:
1. Modify movement distances
2. Add more complex movement patterns
3. Integrate with the full BYOVLA system 