"""PolicyEngine adapter: builds the measured artifact.

Everything in this subpackage requires the ``engine`` extra
(``uv sync --extra engine``) and is only needed to *regenerate*
``democrasim/data/``; the rest of the package runs off the committed
artifact. Imports of the engine stack happen inside functions so the core
package imports cleanly without it.
"""
