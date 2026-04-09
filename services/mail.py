"""邮件发送工具。"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


def _get_smtp_config() -> dict:
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASS", ""),
    }


def _send(to: str, subject: str, html_body: str):
    """发送 HTML 邮件。"""
    cfg = _get_smtp_config()
    if not cfg["user"] or not cfg["password"]:
        logger.warning("SMTP 未配置，跳过邮件发送")
        return

    msg = MIMEMultipart("alternative")
    msg["From"] = cfg["user"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["user"], to, msg.as_string())
        logger.info("邮件已发送至 %s: %s", to, subject)
    except Exception as e:
        logger.error("邮件发送失败 (%s): %s", to, e)


def send_token_email(to: str, token: str, quota: int):
    """向访客发送 Token。"""
    _send(
        to,
        "TechNews Agent - 您的访问 Token",
        f"""
        <div style="font-family:sans-serif; max-width:480px; margin:auto; padding:20px;">
            <h2>TechNews Intelligence Agent</h2>
            <p>您的访问 Token：</p>
            <div style="background:#f4f4f4; padding:12px; border-radius:6px;
                        font-family:monospace; font-size:16px; word-break:break-all;">
                {token}
            </div>
            <p style="color:#666; margin-top:12px;">
                可用次数：{quota} 次。请在网页中粘贴此 Token 开始对话。
            </p>
        </div>
        """,
    )


def send_quota_exhausted_to_admin(admin_email: str, user_email: str, request_id: int, approve_url: str):
    """向管理员发送审批邮件。"""
    _send(
        admin_email,
        f"TechNews Agent - 限额审批请求 ({user_email})",
        f"""
        <div style="font-family:sans-serif; max-width:480px; margin:auto; padding:20px;">
            <h2>限额审批请求</h2>
            <p>用户 <b>{user_email}</b> 已用完初始额度，申请更多对话次数。</p>
            <a href="{approve_url}"
               style="display:inline-block; padding:10px 24px; background:#2563eb;
                      color:#fff; text-decoration:none; border-radius:6px; margin-top:12px;">
                打开审批页
            </a>
            <p style="color:#999; margin-top:16px; font-size:12px;">
                请求 ID: {request_id}
            </p>
        </div>
        """,
    )


def send_quota_exhausted_to_user(to: str):
    """向访客发送等待审批通知。"""
    _send(
        to,
        "TechNews Agent - 额度已用完，等待审批",
        """
        <div style="font-family:sans-serif; max-width:480px; margin:auto; padding:20px;">
            <h2>额度已用完</h2>
            <p>您的初始对话额度已用完。审批请求已自动发送给管理员。</p>
            <p>审批通过后您会收到邮件通知，届时可继续使用。</p>
        </div>
        """,
    )


def send_quota_upgraded(to: str, new_quota: int):
    """向访客发送审批通过通知。"""
    _send(
        to,
        "TechNews Agent - 额度已提升",
        f"""
        <div style="font-family:sans-serif; max-width:480px; margin:auto; padding:20px;">
            <h2>额度已提升</h2>
            <p>您的对话额度已提升至 <b>{new_quota} 次</b>。</p>
        </div>
        """,
    )
