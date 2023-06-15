<?php require_once __DIR__ . '/_header.php'; ?> 

<table>
    <tr>
        <th>Autor</th>
        <th>Naslov</th>
    </tr>

    <?php
        foreach( $bookList as $book )
        {
            echo '<tr>';
            echo '<td>' . $book->author . '</td>';
            echo '<td>' . $book->title . '</td>';
            echo '</tr>';
        }
    ?>
</table>

<?php require_once __DIR__ . '/_footer.php'; ?>
