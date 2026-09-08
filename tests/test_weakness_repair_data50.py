from weakness_repair_data50 import PROFILE, bucket_for, eligible_rows, select_profile_mix, selection_summary


def row(i, domain, language="en", answer="ok answer"):
    h=f"{i:064x}"[-64:]
    g=f"{i+100000:064x}"[-64:]
    return {
        "id":str(i),"prompt":f"prompt {i} unique", "answer":answer,
        "language":language,"domain":domain,"pair_sha256":h,"prompt_group_sha256":g,
        "answer_bytes":len(answer.encode("utf8")),
    }


def test_profile_sums_and_buckets():
    assert abs(sum(PROFILE.values())-1.0)<1e-12
    assert bucket_for({"domain":"math"})=="math"
    assert bucket_for({"domain":"code"})=="code"
    assert bucket_for({"domain":"reasoning"})=="reasoning"
    assert bucket_for({"domain":"format"})=="concise"
    assert bucket_for({"domain":"translation"})=="concise"
    assert bucket_for({"domain":"chat"})=="general"


def test_selection_is_deterministic_and_domain_focused():
    rows=[];i=1
    for domain,n in (("math",500),("code",350),("reasoning",180),("format",180),("chat",240)):
        for j in range(n):
            rows.append(row(i,domain,"de" if j%3==0 else "en"));i+=1
    eligible=eligible_rows(rows,seq=768)
    a=select_profile_mix(eligible,max_examples=1000,min_examples=1000,desired_de_fraction=.35)
    b=select_profile_mix(list(reversed(eligible)),max_examples=1000,min_examples=1000,desired_de_fraction=.35)
    assert [x["pair_sha256"] for x in a]==[x["pair_sha256"] for x in b]
    s=selection_summary(a)
    assert s["examples"]==1000
    assert s["prompt_groups"]==1000
    assert s["buckets"]["math"]>=380
    assert s["buckets"]["code"]>=230
    assert .25<=s["de_fraction"]<=.45


def test_shortage_fills_without_duplicate_groups():
    rows=[];i=1
    for domain,n in (("math",20),("code",20),("reasoning",3),("format",2),("chat",80)):
        for _ in range(n):rows.append(row(i,domain));i+=1
    picked=select_profile_mix(eligible_rows(rows,seq=768),max_examples=100,min_examples=100)
    assert len(picked)==100
    assert len({r["prompt_group_sha256"] for r in picked})==100


def test_multiple_answers_for_same_prompt_group_are_deduplicated_deterministically():
    rows=[row(i,"chat","de" if i%4==0 else "en") for i in range(1,121)]
    duplicate=dict(rows[0])
    duplicate["id"]="duplicate-answer"
    duplicate["pair_sha256"]=f"{999999:064x}"
    duplicate["answer"]="another acceptable answer"
    duplicate["answer_bytes"]=len(duplicate["answer"].encode("utf8"))
    with_duplicate=eligible_rows(rows+[duplicate],seq=768)
    a=select_profile_mix(with_duplicate,max_examples=100,min_examples=100)
    b=select_profile_mix(list(reversed(with_duplicate)),max_examples=100,min_examples=100)
    assert [r["pair_sha256"] for r in a]==[r["pair_sha256"] for r in b]
    assert len(a)==100
    assert len({r["prompt_group_sha256"] for r in a})==100
