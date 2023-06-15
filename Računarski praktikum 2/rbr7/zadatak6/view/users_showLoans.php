<?php require_once __SITE_PATH . '/view/_header.php'; ?>

<h2>Posuđene knjige</h2>
<form method="post" action="<?php echo __SITE_URL . '/index.php?rt=users/returnBook'?>">
<table>
	<tr><th>Autor</th><th>Naslov</th><th>Rok za vraćanje</th><th>Vraćanje knjige</th></tr>
	<?php 
		foreach( $loanedBookList as $book )
		{
			echo '<tr>' .
			     '<td>' . $book['book']->author . '</td>' .
			     '<td>' . $book['book']->title . '</td>' .
			     '<td>' . $book['lease_end'] . '</td>' .
			     '<td><button type="submit" name="loan_id" value="loan_' . $book['loan_id'] . '">Vrati!</button></td>' .
			     '</tr>';
		}
	?>
</table>
</form>


<h2>Posudi novu knjigu</h2>
<form method="post" action="<?php echo __SITE_URL . '/index.php?rt=users/loanBook'?>">
<table>
	<tr><th>Autor</th><th>Naslov</th><th>Posuđivanje knjige</th></tr>
	<?php 
		foreach( $availableBookList as $book )
		{
			echo '<tr>' .
			     '<td>' . $book->author . '</td>' .
			     '<td>' . $book->title . '</td>' .
			     '<td><button type="submit" name="book_id" value="book_' . $book->id . '">Posudi!</button></td>' .
			     '</tr>';
		}
	?>
</table>
</form>

<?php require_once __SITE_PATH . '/view/_footer.php'; ?>
