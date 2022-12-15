import unittest

from classifier.sender import get_sender_judicial_org
from integration.classifier.search_text import label2id


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
    a = "Следственный коммитет РФ"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_judicial_org_4(self):
    a = "ГОВД некого р-на"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  def test_get_sender_v_org_5(self):
    a = "уледственный кАмитет РФ"
    b = get_sender_judicial_org(a)
    self.assertIsNone(b)
