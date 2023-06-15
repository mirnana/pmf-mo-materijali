<?php require_once __DIR__ . '/_header.php'; ?> 

<form action="index.php?rt=books/searchResults" method="post">
    Unesi ime autora:
    <input type="text" name="author">
    <button type="submit">Pretraži bazu!</button>
</form>

<?php require_once __DIR__ . '/_footer.php'; ?>
