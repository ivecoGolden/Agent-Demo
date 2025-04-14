"""add role

Revision ID: 11344d5f90ba
Revises: a0d2c58c292d
Create Date: 2025-04-13 21:15:54.722004
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "11344d5f90ba"
down_revision: Union[str, None] = "a0d2c58c292d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

userrole_enum = sa.Enum("admin", "user", name="userrole")


def upgrade():
    bind = op.get_bind()
    userrole_enum.create(bind, checkfirst=True)
    op.add_column("users", sa.Column("role", userrole_enum, nullable=True))


def downgrade():
    op.drop_column("users", "role")
    userrole_enum.drop(op.get_bind(), checkfirst=True)
