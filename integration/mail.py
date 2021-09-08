import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from socket import gaierror

from analyser.log import logger


def _env_var(vname, default_val=None):
    if vname not in os.environ:
        msg = f'EMAIL : define {vname} environment variable! defaulting to {default_val}'
        logger.warning(msg)
        return default_val
    else:
        return os.environ[vname]


def send_email(smtp_server, port, sender_email, login, password, to, message):
    try:
        context = ssl._create_unverified_context()
        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(login, password)
            server.sendmail(sender_email, to, message)
    except (gaierror, ConnectionRefusedError):
        logger.error('Failed to connect to the server. Bad connection settings?')
    except smtplib.SMTPServerDisconnected:
        logger.error('Failed to connect to the server. Wrong user/password?')
    except smtplib.SMTPException as e:
        logger.error('SMTP error occurred: ' + str(e))


def send_end_audit_email(audit) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_SENDER_EMAIL')
        password = _env_var('GPN_SENDER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server is not None and port is not None and sender_email is not None and password is not None:
            message = MIMEMultipart("alternative")
            message["Subject"] = f"Проверка {audit['subsidiary']['name']} завершена"
            message["From"] = sender_email
            message["To"] = audit['author']['mail']

            if web_url is None:
                text = f"Проверка {audit['subsidiary']['name']} завершена"
                html = f"""\
                <html>
                  <body>
                    <p>
                       Проверка {audit['subsidiary']['name']} завершена 
                    </p>
                  </body>
                </html>
                """
            else:
                audit_url = f"{web_url}/#/audit/result/{str(audit['_id'])}"
                text = f"""\
                Результат доступен по ссылке: {audit_url}"""
                html = f"""\
                <html>
                  <body>
                    <p>
                       <a href="{audit_url}">Проверка {audit['subsidiary']['name']}</a> 
                    </p>
                  </body>
                </html>
                """

            part1 = MIMEText(text, "plain")
            part2 = MIMEText(html, "html")

            message.attach(part1)
            message.attach(part2)

            send_email(smtp_server, port, sender_email, sender_email.split('@')[0], password, audit['author']['mail'], message.as_string())
            return True
    except Exception as e:
        logger.exception(e)
    return False

