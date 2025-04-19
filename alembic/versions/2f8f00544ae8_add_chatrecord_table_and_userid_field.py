"""add ChatRecord table and userid field

Revision ID: 2f8f00544ae8
Revises: 11344d5f90ba
Create Date: 2025-04-16 20:39:13.072548

"""

from typing import Sequence, Union
import uuid
from sqlalchemy.sql import table, column
from sqlalchemy import String

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "2f8f00544ae8"
down_revision: Union[str, None] = "11344d5f90ba"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "chat_record",
        sa.Column("id", sa.Integer(), nullable=False, comment="主键 ID"),
        sa.Column("user_id", sa.Integer(), nullable=True, comment="关联的用户 ID"),
        sa.Column(
            "uuid",
            sa.String(length=64),
            nullable=False,
            comment="前端传递的消息唯一标识",
        ),
        sa.Column(
            "role",
            sa.String(length=32),
            nullable=False,
            comment="消息角色，如 user / assistant",
        ),
        sa.Column(
            "model", sa.String(length=128), nullable=True, comment="使用的大模型名称"
        ),
        sa.Column(
            "response_start_time", sa.DateTime(), nullable=True, comment="回复开始时间"
        ),
        sa.Column(
            "response_end_time", sa.DateTime(), nullable=True, comment="回复结束时间"
        ),
        sa.Column("text", sa.String(), nullable=True, comment="文本内容"),
        sa.Column("image", sa.JSON(), nullable=True, comment="图片列表，字符串数组"),
        sa.Column("video", sa.String(), nullable=True, comment="视频链接"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_record_id"), "chat_record", ["id"], unique=False)
    op.create_index(
        op.f("ix_chat_record_user_id"), "chat_record", ["user_id"], unique=False
    )
    op.create_index(op.f("ix_chat_record_uuid"), "chat_record", ["uuid"], unique=False)
    op.add_column(
        "users",
        sa.Column(
            "userid",
            sa.String(length=64),
            nullable=True,  # 先允许为空
            comment="用户唯一标识（UUID）",
        ),
    )

    users_table = table(
        "users",
        column("id", sa.Integer),
        column("userid", String),
    )

    conn = op.get_bind()
    results = conn.execute(sa.select(users_table.c.id)).fetchall()
    for row in results:
        conn.execute(
            users_table.update()
            .where(users_table.c.id == row.id)
            .values(userid=str(uuid.uuid4()))
        )

    op.alter_column("users", "userid", nullable=False)
    op.create_index(op.f("ix_users_userid"), "users", ["userid"], unique=True)

    op.alter_column(
        "users",
        "id",
        existing_type=sa.INTEGER(),
        comment="主键ID",
        existing_nullable=False,
        autoincrement=True,
    )
    op.alter_column(
        "users",
        "username",
        existing_type=sa.VARCHAR(length=50),
        comment="用户名(唯一)",
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "email",
        existing_type=sa.VARCHAR(length=100),
        comment="邮箱地址(可选)",
        existing_nullable=True,
    )
    op.alter_column(
        "users",
        "hashed_password",
        existing_type=sa.VARCHAR(length=128),
        comment="加密密码(必填)",
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "is_active",
        existing_type=sa.BOOLEAN(),
        comment="账号是否激活",
        existing_nullable=True,
    )
    op.alter_column(
        "users",
        "created_at",
        existing_type=postgresql.TIMESTAMP(timezone=True),
        comment="创建时间(自动设置)",
        existing_nullable=True,
        existing_server_default=sa.text("now()"),
    )
    op.alter_column(
        "users",
        "role",
        existing_type=postgresql.ENUM("admin", "user", name="userrole"),
        comment="用户角色(默认普通用户)",
        existing_nullable=True,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_users_userid"), table_name="users")
    op.alter_column(
        "users",
        "role",
        existing_type=postgresql.ENUM("admin", "user", name="userrole"),
        comment=None,
        existing_comment="用户角色(默认普通用户)",
        existing_nullable=True,
    )
    op.alter_column(
        "users",
        "created_at",
        existing_type=postgresql.TIMESTAMP(timezone=True),
        comment=None,
        existing_comment="创建时间(自动设置)",
        existing_nullable=True,
        existing_server_default=sa.text("now()"),
    )
    op.alter_column(
        "users",
        "is_active",
        existing_type=sa.BOOLEAN(),
        comment=None,
        existing_comment="账号是否激活",
        existing_nullable=True,
    )
    op.alter_column(
        "users",
        "hashed_password",
        existing_type=sa.VARCHAR(length=128),
        comment=None,
        existing_comment="加密密码(必填)",
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "email",
        existing_type=sa.VARCHAR(length=100),
        comment=None,
        existing_comment="邮箱地址(可选)",
        existing_nullable=True,
    )
    op.alter_column(
        "users",
        "username",
        existing_type=sa.VARCHAR(length=50),
        comment=None,
        existing_comment="用户名(唯一)",
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "id",
        existing_type=sa.INTEGER(),
        comment=None,
        existing_comment="主键ID",
        existing_nullable=False,
        autoincrement=True,
    )
    op.drop_column("users", "userid")
    op.drop_index(op.f("ix_chat_record_uuid"), table_name="chat_record")
    op.drop_index(op.f("ix_chat_record_user_id"), table_name="chat_record")
    op.drop_index(op.f("ix_chat_record_id"), table_name="chat_record")
    op.drop_table("chat_record")
    # ### end Alembic commands ###
