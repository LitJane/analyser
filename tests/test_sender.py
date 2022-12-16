import unittest

from integration.classifier.sender import  get_sender_judicial_org


class ClassifierTestCase(unittest.TestCase):
  def test_get_sender_judicial_org_1(self):
    a = "Некий отправитель типа Следственный комитет РФ"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_judicial_org_0(self):
    a = "Некий отправитель типа Наследственный комитет РФ"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_v_org_2(self):
    a = "Следственный кАмитет РФ"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_judicial_org_3(self):
    a = "Следственный коммитет РФ <komitet@russia.ru>"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_judicial_org_4(self):
    a = "ГОВД некого р-на <komitet@russia.ru>"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_v_org_5(self):
    a = "уледственный кАмитет РФ"
    b = get_sender_judicial_org(a)
    self.assertIsNone(b)

  def test_get_sender_judicial_org_6(self):
    a = "ГЛАВНОЕ УПРАВЛЕНИЕ МИНИСТЕРСТВА ВНУТРЕННИХ ДЕЛ РОССИЙСКОЙ ФЕДЕРАЦИИ ПО ГОРОДУ МОСКВЕ (ГУ МВД России по г. Москве) Дежурная часть"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)
