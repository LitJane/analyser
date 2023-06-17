import unittest

from bson import ObjectId

from analyser import finalizer
from analyser.runner import apply_judical_practice
from integration.classifier.sender import get_sender_judicial_org
from tests.test_utilits import NO_DB, NO_DB_ERR_MSG


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
    a = "СУ СК России по Ямало- Ненецкому автономному округу Ноябрьский межрайонный следственный отдел <zaborskiy.test@proton.me>"
    b = get_sender_judicial_org(a)
    self.assertIsNotNone(b)

  @unittest.skipIf(NO_DB, NO_DB_ERR_MSG)
  def test_get_sender_judicial_org_from_header(self):
    document = finalizer.get_doc_by_id(ObjectId("639c64de4b01c8adaa5a4f61"))
    document = document['parse']
    print(document)

    headline = document['paragraphs'][0]['paragraphHeader']['text']
    print(headline)
    b = get_sender_judicial_org(headline)
    print(b)

    self.assertIsNotNone(b)

  def test_apply_judical_practice(self):
    classification_result = None
    classification_result = apply_judical_practice(classification_result, "sender_judicial_org")
    print(classification_result)
    self.assertEqual("sender_judicial_org", classification_result[0]['sender_judicial_org'])
    self.assertEqual("Практика судебной защиты", classification_result[0]['label'])
