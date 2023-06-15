<?php require_once __SITE_PATH . '/view/_header.php'; ?>

<form method="post" action="<?php echo __SITE_URL . '/index.php?rt=users/showLoans'?>">
	<h2>Popis svih korisnika:</h2>
	<table>
		<tr><th>Ime</th><th>Prezime</th><th>Posuđivanje knjiga</th></tr>
		<?php 
			foreach( $userList as $user )
			{
				echo '<tr>' .
				     '<td>' . $user->surname . '</td>' .
				     '<td>' . $user->name . '</td>' .
				     '<td><button type="submit" name="user_id" value="user_' . $user->id . '">Vidi!</button></td>' .
				     '</tr>';
			}
		?>
	</table>
</form>

<?php require_once __SITE_PATH . '/view/_footer.php'; ?>
