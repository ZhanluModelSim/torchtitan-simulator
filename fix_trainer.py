import re

with open('torchtitan/experiments/simulator/trainer.py', 'r') as f:
    content = f.read()

# Replace `return None, model_parts, True, True` with a mock schedule object
content = content.replace('    return None, model_parts, True, True', '''
    class MockSchedule:
        def step(self, *args, **kwargs):
            return torch.tensor(0.0, device="meta")

    return MockSchedule(), model_parts, True, True
''')

with open('torchtitan/experiments/simulator/trainer.py', 'w') as f:
    f.write(content)
