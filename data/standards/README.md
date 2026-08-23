# Acceptance criteria — sample documents

These are **original documents written for this project**, not reproductions of
any published standard. IPC-A-610 and its equivalents are copyrighted and are
deliberately not included here.

They are written the way a plant's internal work instructions are written, so
the retrieval tool has realistic material to search, but they carry no
authority and must not be used to judge a real board.

## Front matter

Every document declares the class it governs:

```
---
defect_class: open
---
```

`defect_class: any` marks a document that governs every class -- QP-110's
escape budget and WI-300's operating procedure. The declaration is carried onto
every passage cut from the file and is what scopes retrieval, so a document
that omits it fails the index build rather than quietly becoming evidence about
every class. A new class is added by writing its work instruction with the
class named here; nothing else has a list to update.
