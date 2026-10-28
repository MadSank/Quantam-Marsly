import { useEffect, useRef, useState } from 'react'

/**
 * A polite live region that reads out `message` when it changes (a new site, class, model or
 * seed). The first message is not announced: on page load the heading already says it.
 */
export function Announcer({ message }: { message: string }) {
  const first = useRef(message)
  const [text, setText] = useState('')
  useEffect(() => {
    if (message === first.current) return
    first.current = ''
    // Set after a tick so a repeated change still reaches screen readers as new text.
    setText('')
    const t = window.setTimeout(() => setText(message), 50)
    return () => window.clearTimeout(t)
  }, [message])
  return (
    <p className="visually-hidden" role="status" aria-live="polite" data-announcer>
      {text}
    </p>
  )
}
