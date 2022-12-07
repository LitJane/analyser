import os
import smtplib
import ssl
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from socket import gaierror
from urllib.parse import urljoin

from analyser.log import logger


def _env_var(vname, default_val=None):
    if vname not in os.environ:
        msg = f'EMAIL : define {vname} environment variable! defaulting to {default_val}'
        logger.warning(msg)
        return default_val
    else:
        return os.environ[vname]


def send_email(smtp_server, port, login, password, message):
    try:
        context = ssl._create_unverified_context()
        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(login, password)
            server.send_message(message)
    except (gaierror, ConnectionRefusedError):
        logger.error('Failed to connect to the server. Bad connection settings?')
    except smtplib.SMTPServerDisconnected:
        logger.error('Failed to connect to the server. Wrong user/password?')
    except smtplib.SMTPException as e:
        logger.error('SMTP error occurred: ' + str(e))


def escape_email_headers(input: str) -> str:
    return input.replace('\r', '').replace('\n', '')


def send_end_audit_email(audit) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_SENDER_EMAIL')
        login = _env_var('GPN_SENDER_LOGIN')
        password = _env_var('GPN_SENDER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server is not None and port is not None and sender_email is not None and password is not None and login:
            message = MIMEMultipart("alternative")
            message["Subject"] = escape_email_headers(f"Проверка {audit['subsidiary']['name']} завершена")
            message["From"] = escape_email_headers(sender_email)
            message["To"] = escape_email_headers(audit['author']['mail'])

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

            send_email(smtp_server, port, login, password, message)
            return True
    except Exception as e:
        logger.exception(e)
    return False


def generate_links(audit, practices: [], web_url) -> str:
    url = urljoin(web_url, 'user-practice')
    result = ""
    for practice in practices:
        result += f"""<p><a href="{url + '?audit_id=' + str(audit['_id']) + '&practice_id=' + str(practice['_id'])}">{practice['label']}</a></p>"""
    return result


def send_classifier_email(audit, top_classification_result, attachments: [], practices) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_CLASSIFIER_EMAIL')
        login = _env_var('GPN_CLASSIFIER_LOGIN')
        password = _env_var('GPN_CLASSIFIER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server and port and sender_email and password and login and web_url:
            message = EmailMessage()
            message["Subject"] = escape_email_headers(f"{audit['additionalFields']['email_subject']}")
            message["From"] = escape_email_headers(sender_email)
            message["To"] = escape_email_headers(top_classification_result['email'])
            message['Cc'] = escape_email_headers(audit['additionalFields']['email_from'])

            plain_text = f"""
                Здравствуйте!
                
                Направляем вам результат определения юридической практики для высланного ранее документа.
                
                Результат: {top_classification_result['label']}
                """

            html = f"""\
                <html>
                  <body>
                    <p>
                        Здравствуйте!
                    </p>
                    <p>
                        Направляем вам результат определения юридической практики для высланного ранее документа.
                    </p>
                    <p style="font-size: 200%">
                        Результат: {top_classification_result['label']}
                    </p>
                    <p>
                        <b>Если юридическая практика для документа определена неверно, то кликните на верную юридическую практику:</b>
                    </p>
                    {generate_links(audit, practices, web_url)}
                  </body>
                </html>
                """

            message.set_content(plain_text)
            message.add_alternative(html, subtype='html')
            for attachment in attachments:
                maintype, subtype = attachment.ctype.split('/', 1)
                message.add_attachment(attachment.read(), filename=attachment.filename, maintype=maintype, subtype=subtype)

            send_email(smtp_server, port, login, password, message)
            return True
    except Exception as e:
        logger.exception(e)
    return False


def generate_errors(errors, html) -> str:
    result = ""
    if html:
        for error in errors:
            result += f"""<p>Ошибка: {error['text']}</p>"""
    else:
        for error in errors:
            result += f"""Ошибка: {error['text']}"""
    return result


def send_classifier_error_email(audit, attachments: []) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_CLASSIFIER_EMAIL')
        login = _env_var('GPN_CLASSIFIER_LOGIN')
        password = _env_var('GPN_CLASSIFIER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server and port and sender_email and password and login and web_url:
            message = EmailMessage()
            message["Subject"] = escape_email_headers(f"{audit['additionalFields']['email_subject']}")
            message["From"] = escape_email_headers(sender_email)
            message['To'] = escape_email_headers(audit['additionalFields']['email_from'])

            plain_text = f"""
                Здравствуйте!
                
                К сожалению, определить юридическую практику для высланного ранее документа не удалось.
                
                {generate_errors(audit['errors'], False)}
                """

            html = f"""\
                <html>
                  <body>
                    <p>
                        Здравствуйте!
                    </p>
                    <p>
                        К сожалению, определить юридическую практику для высланного ранее документа не удалось.
                    </p>
                    {generate_errors(audit['errors'], True)}
                  </body>
                </html>
                """

            message.set_content(plain_text)
            message.add_alternative(html, subtype='html')
            for attachment in attachments:
                maintype, subtype = attachment.ctype.split('/', 1)
                message.add_attachment(attachment.read(), filename=attachment.filename, maintype=maintype, subtype=subtype)

            send_email(smtp_server, port, login, password, message)
            return True
    except Exception as e:
        logger.exception(e)
    return False


def send_compliance_info_email(audit) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_CLASSIFIER_EMAIL')
        login = _env_var('GPN_CLASSIFIER_LOGIN')
        password = _env_var('GPN_CLASSIFIER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server and port and sender_email and password and login and web_url:
            message = EmailMessage()
            message["Subject"] = f"{audit['additionalFields']['email_subject']}"
            message["From"] = sender_email
            message['To'] = audit['additionalFields']['email_from']

            plain_text = f"""
                Здравствуйте!
                
                Документ {audit['additionalFields']['main_filename']} находится на рассмотрении экспертов Практики корпоративного мониторинга и комплаенса на предмет необходимости одобрения сделки.
                """

            html = f"""\
                <html>
                  <body>
                    <p>
                        Здравствуйте!
                    </p>
                    <p>
                        Документ {audit['additionalFields']['main_filename']} находится на рассмотрении экспертов Практики корпоративного мониторинга и комплаенса на предмет необходимости одобрения сделки.
                    </p>
                  </body>
                </html>
                """

            message.set_content(plain_text)
            message.add_alternative(html, subtype='html')
            send_email(smtp_server, port, login, password, message)
            return True
    except Exception as e:
        logger.exception(e)
    return False


def send_compliance_error_email(audit, errors, to) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_CLASSIFIER_EMAIL')
        login = _env_var('GPN_CLASSIFIER_LOGIN')
        password = _env_var('GPN_CLASSIFIER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server and port and sender_email and password and login and web_url:
            message = EmailMessage()
            message["Subject"] = f"{audit['additionalFields']['email_subject']} [Ошибка анализа договорного документа]"
            message["From"] = sender_email
            message['To'] = to

            document_url = f"{web_url}/#/classifier/edit/{str(audit['additionalFields']['main_document_id'])}"
            plain_text = f"""
                Здравствуйте!
                
                В модуль Классификатор для проверки на необходимость одобрения коллегиаль-ным органом загружен договорной документ, при анализе которого обнаружена(-ы) ошибка(-и).
                
                Ссылка на документ: {document_url} 
                
                {generate_errors(errors, False)}
                """

            html = f"""\
                <html>
                  <body>
                    <p>
                        Здравствуйте!
                    </p>
                    <p>
                        В модуль Классификатор для проверки на необходимость одобрения коллегиаль-ным органом загружен договорной документ, при анализе которого обнаружена(-ы) ошибка(-и).
                    </p>
                    <p>
                        Ссылка на документ: {document_url} 
                    </p>
                    {generate_errors(errors, True)}
                  </body>
                </html>
                """

            message.set_content(plain_text)
            message.add_alternative(html, subtype='html')
            send_email(smtp_server, port, login, password, message)
            return True
    except Exception as e:
        logger.exception(e)
    return False


def send_compliance_protocol_praparation_email(audit, ) -> bool:
    try:
        smtp_server = _env_var('GPN_SMTP_SERVER')
        port = _env_var('GPN_SMTP_PORT')
        sender_email = _env_var('GPN_CLASSIFIER_EMAIL')
        login = _env_var('GPN_CLASSIFIER_LOGIN')
        password = _env_var('GPN_CLASSIFIER_PASSWORD')
        web_url = _env_var('GPN_WEB_URL')

        if smtp_server and port and sender_email and password and login and web_url:
            message = EmailMessage()
            message["Subject"] = f"{audit['additionalFields']['email_subject']}"
            message["From"] = sender_email
            message['To'] = audit['additionalFields']['email_from']

            plain_text = f"""
                Здравствуйте!
                
                Сделка согласно договорному документу {audit['additionalFields']['main_filename']} требует одобрения коллегиальным органом. Экспертами Практики корпоративного мониторинга и комплаенса готовится протокол одобрения.
                """

            html = f"""\
                <html>
                  <body>
                    <p>
                        Здравствуйте!
                    </p>
                    <p>
                        Сделка согласно договорному документу {audit['additionalFields']['main_filename']} требует одобрения коллегиальным органом. Экспертами Практики корпоративного мониторинга и комплаенса готовится протокол одобрения.
                    </p>
                  </body>
                </html>
                """

            message.set_content(plain_text)
            message.add_alternative(html, subtype='html')
            send_email(smtp_server, port, login, password, message)
            return True
    except Exception as e:
        logger.exception(e)
    return False