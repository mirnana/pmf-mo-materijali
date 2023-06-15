<?php require_once __DIR__ . '/_header.php'; ?>

<table>
	<tr><th>Prezime</th><th>Ime</th></tr>
	<?php 
		foreach( $userList as $user )
		{
			echo '<tr>' .
			     '<td>' . $user->surname . '</td>' .
			     '<td>' . $user->name . '</td>' .
			     '</tr>';
		}
	?>
</table>

<?php require_once __DIR__ . '/_footer.php'; ?>
