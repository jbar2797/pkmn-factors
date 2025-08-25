"""merge heads 7b9e2c8a5c2e d4d8a5a5044e

Revision ID: 14adebb5f39a
Revises: 7b9e2c8a5c2e, d4d8a5a5044e
Create Date: 2025-08-24 17:12:40.707571

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "14adebb5f39a"
down_revision: Union[str, Sequence[str], None] = ("7b9e2c8a5c2e", "d4d8a5a5044e")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
