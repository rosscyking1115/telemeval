# Security Policy

## Supported versions

telemeval is pre-1.0; security fixes are made against the latest released
version on PyPI.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to **rosscyking@gmail.com**
rather than opening a public issue. Include a description, affected version,
and a minimal reproduction if possible. You can expect an acknowledgement
within a few days.

telemeval's runtime is a pure numpy/pandas library with no network, file
execution, or credential handling, so its attack surface is small; the most
likely relevant reports concern the input-parsing contract or the build/
release supply chain.
