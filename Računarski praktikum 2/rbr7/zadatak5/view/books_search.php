<?php require_once __DIR__ . '/_header.php'; ?>

<form method="post" action="index.php?rt=books/searchResults">
	Unesi ime autora čije te knjige zanimaju:
	<input type="text" name="author" />

	<button type="submit">Traži</button>
</form>

<?php require_once __DIR__ . '/_footer.php'; ?>
