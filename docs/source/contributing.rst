Contributing
============

AtomVoxelizer is primarily managed by `Tristan Maxson
<https://www.linkedin.com/in/tgmaxson/>`_. Open contributions are welcome,
especially focused bug fixes, documentation improvements, tests, and small
features that fit the existing API.

Project Links
-------------

Use these links when working with the project:

.. list-table::
   :header-rows: 1

   * - Resource
     - Link
   * - Source repository and merge requests
     - https://gitlab.com/tgmaxson/atomvoxelizer
   * - Issues
     - https://gitlab.com/tgmaxson/atomvoxelizer/-/issues
   * - PyPI package
     - https://pypi.org/project/AtomVoxelizer/
   * - Documentation
     - https://atomvoxelizer.readthedocs.io/en/latest/index.html
   * - Zenodo record
     - https://zenodo.org/records/15479550

Reporting Issues
----------------

Open an issue on GitLab for bugs, confusing behavior, documentation problems,
or feature proposals. A useful issue includes:

* Operating system
* Python version
* AtomVoxelizer version
* Minimal reproducible example
* Complete error traceback, if there is one

For larger changes, open an issue first so the scope can be discussed before
implementation work begins.

Development Setup
-----------------

Clone the GitLab repository and install the package in editable mode with the
development dependencies:

.. code-block:: bash

   git clone https://gitlab.com/tgmaxson/atomvoxelizer.git
   cd atomvoxelizer
   pip install -e ".[dev]"

Run the test suite before submitting a merge request:

.. code-block:: bash

   pytest tests/

Build the documentation locally when changing documentation or public APIs:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Contribution Guidelines
-----------------------

Please keep merge requests focused on one bug fix, feature, or documentation
change. Clear, small changes are easier to review and maintain.

Code changes should follow the existing style in the repository. Add tests for
new behavior, update documentation when user-facing behavior changes, and avoid
adding required dependencies unless they are necessary for the core NumPy
backend. Optional acceleration and analysis packages should stay optional.

All submitted code is reviewed before acceptance. Review may cover correctness,
maintainability, tests, documentation, dependency choices, and consistency with
the package API. Contributors should expect requested revisions before a merge
request is accepted.

AI-assisted development tools, including tools such as Claude Code and Codex,
are generally permitted. Contributors remain responsible for the submitted
code, tests, documentation, and licensing. AI-assisted changes should follow
the same style and quality expectations as any other contribution, and should
not introduce unrelated rewrites, unexplained complexity, or generated code
that cannot be reviewed and maintained.

Merge Requests
--------------

Before opening a merge request:

* Rebase or merge the latest ``main`` branch.
* Run ``pytest tests/``.
* Build the documentation if docs or API behavior changed.
* Write a short description of the change and the verification performed.

Contributors
------------

This list can be updated as people contribute code, tests, documentation,
examples, or review:

.. list-table::
   :header-rows: 1

   * - Contributor
     - Role
   * - `Tristan Maxson <https://www.linkedin.com/in/tgmaxson/>`_
     - Primary maintainer
