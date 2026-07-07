---
name: Bug report
about: Something is wrong (a crash, a wrong number, a contract error that shouldn't be)
labels: bug
---

**telemeval version** (`python -c "import telemeval; print(telemeval.__version__)"`):

**What happened / what did you expect?**

**Minimal reproduction** (labels + predictions small enough to paste):

```python
import pandas as pd
from telemeval import evaluate

labels = pd.DataFrame({...})
predictions = {"channel": pd.DataFrame({...})}
result = evaluate(labels, predictions)
```

**If you believe a metric value is wrong:** what reference (paper, other
implementation, hand computation) gives a different number, and what number?
