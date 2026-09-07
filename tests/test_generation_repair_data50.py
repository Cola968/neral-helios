from generation_repair_data50 import eligible_rows, formatted_training_tokens, select_language_mix, selection_digest


def row(i,lang="en",answer="A useful short answer."):
    return {
        "id":str(i),"language":lang,"domain":"chat","prompt":f"Question {i} with enough text?","answer":answer,
        "answer_bytes":len(answer.encode("utf8")),"pair_sha256":f"{i+1:064x}","prompt_group_sha256":f"{i+1000:064x}",
    }


def test_formatted_length_accounts_for_specials_and_bytes():
    r=row(1,answer="ä")
    expected=1+len(("User: "+r["prompt"]+"\nAssistant:").encode("utf8"))+len(" ä\n".encode("utf8"))+1
    assert formatted_training_tokens(r)==expected


def test_eligible_rows_enforces_answer_and_sequence_bounds():
    good=row(1,answer="This is within bounds.")
    tiny=row(2,answer="x")
    huge=row(3,answer="x"*500)
    chosen=eligible_rows([good,tiny,huge],seq=256,min_answer_bytes=4,max_answer_bytes=100)
    assert [r["id"] for r in chosen]==["1"]


def test_language_mix_is_deterministic_and_prefers_requested_de_share():
    rows=[row(i,"de" if i<30 else "en") for i in range(100)]
    a=select_language_mix(rows,max_examples=50,desired_de_fraction=.20,min_examples=10)
    b=select_language_mix(list(reversed(rows)),max_examples=50,desired_de_fraction=.20,min_examples=10)
    assert [r["pair_sha256"] for r in a]==[r["pair_sha256"] for r in b]
    assert sum(r["language"]=="de" for r in a)==10
    assert selection_digest(a)==selection_digest(b)


def test_language_mix_fills_shortage_from_other_language():
    rows=[row(i,"de") for i in range(2)]+[row(i+100,"en") for i in range(20)]
    chosen=select_language_mix(rows,max_examples=10,desired_de_fraction=.5,min_examples=5)
    assert len(chosen)==10
    assert sum(r["language"]=="de" for r in chosen)==2
