<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Zadatak 1</title>
	<style>table, th, td { border: solid 1px; }</style>
</head>
<body>
<?php
	$n = 10;
?>

	<table>
		<tr>
			<td>*</td>
			<?php 
				for( $c = 1; $c <= $n; ++$c )
					echo "<th>$c</th>";
			?>
		</tr>
		<?php 
			for( $r = 1; $r <= $n; ++$r )
			{
				echo "<tr><th>$r</th>";
				for( $c = 1; $c <= $n; ++$c )
					echo "<td>" . $r * $c . "</td>";
				echo "</tr>";
			}
		?>
	</table>
</body>
</html> 
